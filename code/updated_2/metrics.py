# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Scoring for the clutter experiments.

The metric that matters here is Leak%, which the earlier benches did not have.

  Extra%    fraction of predicted surface further than `thresh` x extent from the
            GT target surface. This is what `mask_guided/score_heldout.py` reports.
            It counts ALL surplus geometry — genuine hallucination, shape-proportion
            error, and leaked clutter are indistinguishable inside it. On the woven
            stool most of a 55% Extra score turned out to be fidelity error with no
            background object present at all, so Extra% alone cannot support a claim
            about clutter.

  Leak%     fraction of predicted surface that is nearer to the PLACED DISTRACTOR
            than to the target, and genuinely close to it. Because `clutter.py`
            writes the distractor's exact placed mesh, this is a direct measurement
            of the paper's Limitation (b) rather than an inference from Extra%.

  Missing%  fraction of GT target surface with no predicted surface nearby — the
            direction that catches a suppression method cutting into the object.

  Chamfer%  symmetric mean surface distance as % of target extent.

Extra%/Chamfer% follow the existing convention (`thresh=0.03`, 8000 samples, extent
= max GT bounding-box side) so numbers stay comparable with the earlier runs.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

N_SAMPLES = 8000
THRESH = 0.03


def _sample(mesh, n, seed=0):
    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(1 << 30)))
    return np.asarray(pts, np.float64)


def load_label(npz_path):
    """Return (target_mesh, distractor_mesh) from a clutter label file."""
    d = np.load(npz_path)
    tgt = trimesh.Trimesh(vertices=d["target_vertices"].astype(np.float64),
                          faces=d["target_faces"], process=False)
    dis = trimesh.Trimesh(vertices=d["distractor_vertices"].astype(np.float64),
                          faces=d["distractor_faces"], process=False)
    return tgt, dis


def score(pred, target, distractor=None, thresh=THRESH, n=N_SAMPLES, seed=0):
    """
    Score one predicted mesh. `pred`/`target`/`distractor` are trimesh objects in
    the model frame at metric scale. `distractor` may be None (clean condition), in
    which case Leak% is not reported.
    """
    if len(pred.faces) == 0 or len(target.faces) == 0:
        return None

    pp = _sample(pred, n, seed)
    gp = _sample(target, n, seed)
    ext = float(np.max(target.extents))
    tol = thresh * ext

    d_pred, _ = cKDTree(gp).query(pp, k=1)      # pred -> gt   (extra geometry)
    d_gt, _ = cKDTree(pp).query(gp, k=1)        # gt   -> pred (missing geometry)

    out = {
        "extra_pct": float(np.mean(d_pred > tol)) * 100,
        "missing_pct": float(np.mean(d_gt > tol)) * 100,
        "chamfer_pct": float((d_pred.mean() + d_gt.mean()) / 2 / ext) * 100,
        "extent": ext,
        "n_pred_faces": int(len(pred.faces)),
    }

    if distractor is not None and len(distractor.faces):
        dp = _sample(distractor, n, seed)
        d_to_dist, _ = cKDTree(dp).query(pp, k=1)
        # Leaked surface: closer to the distractor than to the target, and actually
        # near it. Both conditions matter — the first alone would count surface that
        # is merely on the far side of the object, the second alone would count the
        # contact region the two share.
        leaked = (d_to_dist < d_pred) & (d_to_dist < tol)
        out["leak_pct"] = float(np.mean(leaked)) * 100
        # How much of the distractor got reproduced, i.e. coverage of the intruder.
        d_dist_to_pred, _ = cKDTree(pp).query(dp, k=1)
        out["distractor_recall_pct"] = float(np.mean(d_dist_to_pred < tol)) * 100
    return out


def score_paths(pred_path, label_npz=None, target_mesh=None, **kw):
    """Convenience wrapper taking file paths."""
    from pathlib import Path
    if not Path(pred_path).exists():
        return None
    pred = trimesh.load(pred_path, force="mesh")
    if label_npz is not None:
        tgt, dis = load_label(label_npz)
    else:
        tgt, dis = target_mesh, None
    return score(pred, tgt, dis, **kw)


def aggregate(rows, keys=("extra_pct", "leak_pct", "missing_pct", "chamfer_pct")):
    """Mean of each metric over rows that have it."""
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r and k in r]
        if vals:
            out[k] = float(np.mean(vals))
            out[k + "_std"] = float(np.std(vals))
    out["n"] = len([r for r in rows if r])
    return out


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """
    Paired difference (b - a) with a bootstrap 95% CI.

    The multi-seed verdict on the earlier mask-guided work turned on exactly this:
    single-seed deltas looked convincing and did not survive. Any claim made from
    these experiments should quote the CI, not the mean alone.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert len(a) == len(b) and len(a) > 0
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    n_pos = int((d > 0).sum())
    # With one or two pairs the resample has (almost) no variance and the CI
    # collapses onto the mean, which would report every difference as significant.
    # Significance is only claimed from three pairs up.
    sig = bool((lo > 0 or hi < 0) and len(d) >= 3)
    return {"mean_diff": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_pairs": len(d), "n_positive": n_pos, "significant": sig}
