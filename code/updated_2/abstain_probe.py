# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Can we TELL whether an intruder leaked in, without ground truth?

This is the one thing blocking the whole method. Stage 2/3 showed `refeed` removes leakage at
no damage on affected objects — but on UNAFFECTED objects every method still degrades the
reconstruction. So a system that always acts is not usable. It needs to abstain.

Abstention is a detection problem, and it can be measured on meshes that already exist:
150 Stage-1 runs, half clean and half cluttered, with Leak% known for every one. Zero GPU.

The detector may use ONLY what inference has — the reconstructed mesh and the SLAM cloud:

    unsupported   share of predicted surface with no observed point nearby. Leaked geometry sits
                  where nothing was measured. On its own this also fires on ordinary unobserved
                  surface, which is exactly why it is tested rather than assumed.
    top_unsup     the same, restricted to surface in the upper part of the object. Things get
                  stacked ON tables, so an intruder is up top; a table's unmeasured underside
                  is not.
    blob_frac     largest connected chunk of unsupported surface, as a share of the whole. An
                  intruder is one lump; ordinary reconstruction error is scattered.

Reported as ROC AUC (1.0 = perfect, 0.5 = coin flip) against the true clean/cluttered label.

    python updated_2/abstain_probe.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
S1 = REPO / "updated_2/outputs/stage1/results.json"


def cloud_for(name):
    import pickle
    d = pickle.load(open(REPO / f"data/{name}.pkl", "rb"))
    return d["points_model"].numpy().astype(np.float64)[:, :3]


def detector_scores(mesh_path, cloud, tau_frac=0.03, n=12000, seed=0):
    m = trimesh.load(mesh_path, force="mesh")
    if len(getattr(m, "faces", [])) == 0:
        return None
    scale = float(np.max(np.ptp(cloud, axis=0)))
    tau = tau_frac * scale
    pts, _ = trimesh.sample.sample_surface(m, n, seed=seed)
    d_pred, _ = cKDTree(cloud).query(pts, k=1)
    unsup = d_pred > tau
    out = {"unsupported": float(np.mean(unsup))}

    # "upper part" is defined by the CLOUD's own vertical range, not the mesh's, so a bloated
    # mesh cannot move the boundary and flatter its own score.
    zlo, zhi = float(np.min(cloud[:, 2])), float(np.max(cloud[:, 2]))
    cut = zlo + 0.72 * (zhi - zlo)
    top = pts[:, 2] > cut
    out["top_unsup"] = float(np.mean(unsup & top))

    # largest connected lump of unsupported surface, by single-linkage within 4% of extent
    idx = np.where(unsup)[0]
    if len(idx) < 8:
        out["blob_frac"] = 0.0
    else:
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import coo_matrix
        P = pts[idx]
        pairs = cKDTree(P).query_pairs(0.04 * scale, output_type="ndarray")
        if len(pairs) == 0:
            out["blob_frac"] = 1.0 / len(pts)
        else:
            g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
            ncomp, lab = connected_components(g, directed=False)
            big = np.bincount(lab).max()
            out["blob_frac"] = float(big / len(pts))
    return out


def auc(scores, labels):
    """Rank-based ROC AUC — the probability a cluttered run scores above a clean one."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    rows = json.loads(S1.read_text())
    clouds = {}
    recs = []
    for r in rows:
        p = REPO / r["mesh"]
        if not p.exists():
            continue
        if r["name"] not in clouds:
            clouds[r["name"]] = cloud_for(r["name"])
        s = detector_scores(str(p), clouds[r["name"]])
        if s is None:
            continue
        s.update(name=r["name"], mode=r["mode"], level=r["level"], seed=r["seed"],
                 condition=r["condition"], leak=r["leak_pct"])
        recs.append(s)
        if len(recs) % 30 == 0:
            print(f"  scored {len(recs)}")

    feats = ["unsupported", "top_unsup", "blob_frac"]
    y = [1 if r["condition"] == "clutter" else 0 for r in recs]
    print(f"\nscored {len(recs)} runs ({sum(y)} cluttered / {len(y)-sum(y)} clean)\n")
    print(f"{'detector':16}{'AUC all':>10}", end="")
    levels = sorted({r["level"] for r in recs})
    for L in levels:
        print(f"{('L='+str(L)):>10}", end="")
    print()
    for f in feats:
        line = f"{f:16}{auc([r[f] for r in recs], y):10.3f}"
        for L in levels:
            sub = [r for r in recs if r["level"] == L]
            line += f"{auc([r[f] for r in sub], [1 if r['condition']=='clutter' else 0 for r in sub]):10.3f}"
        print(line)

    # a detector is only useful if it fires on the objects that actually leak
    print("\nper-object AUC (top_unsup), with mean true leak rise:")
    for nm in sorted({r["name"] for r in recs}):
        sub = [r for r in recs if r["name"] == nm]
        ys = [1 if r["condition"] == "clutter" else 0 for r in sub]
        dl = np.mean([r["leak"] for r in sub if r["condition"] == "clutter"]) - \
             np.mean([r["leak"] for r in sub if r["condition"] == "clean"])
        print(f"  {nm:28} AUC {auc([r['top_unsup'] for r in sub], ys):5.3f}   true dLeak {dl:+6.2f}")

    json.dump(recs, open(REPO / "updated_2/outputs/abstain_probe.json", "w"), indent=2)
    print(f"\nwrote updated_2/outputs/abstain_probe.json")


if __name__ == "__main__":
    main()
