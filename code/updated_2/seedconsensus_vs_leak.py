# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Does v2 seed consensus help or HURT the extra-object problem?

v2 picks, out of K draws, the one that best agrees with the object's own SLAM points. It was
validated on Extra% with clean inputs. This asks a different question it was never tested on:
when a second object is standing on the table, does choosing this way leave LESS of that object
in the mesh, or more?

There is a concrete reason to expect it to BACKFIRE. In `points` and `both` clutter the
intruder's points are IN the cloud the selector trusts, so a draw that faithfully reconstructs
the intruder agrees with the cloud BETTER. The selector would then prefer exactly the draws that
leak. In `image` clutter the intruder never enters the cloud, so the same rule should push the
other way.

That split is the test. If leakage rises in `points`/`both` and falls in `image`, the mechanism
is understood. If it moves the same way everywhere, something else is going on.

Uses the 150 Stage-1 meshes already on disk — 3 seeds per cell, so K=3. No GPU.

    python updated_2/seedconsensus_vs_leak.py
"""

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "updated_2"))
CLUT = REPO / "updated_2/outputs/stage1/clutter"


def cloud_seen_at_inference(obj, mode, level, condition):
    """
    The cloud the SELECTOR would actually see.

    For clean runs, and for `image` clutter, that is the original cloud — image-only clutter
    never touches the points. For `points`/`both` it is the CLUTTERED cloud, intruder included.
    Getting this wrong would hand the selector information it does not have at inference.
    """
    if condition == "clean" or mode == "image":
        p = REPO / f"data/{obj}.pkl"
    else:
        tag = "L" + str(level).replace("0.", "0")
        p = CLUT / f"{obj}__{mode}__{tag}.pkl"
        if not p.exists():
            p = REPO / f"data/{obj}.pkl"
    d = pickle.load(open(p, "rb"))
    return d["points_model"].numpy().astype(np.float64)[:, :3]


def cloud_fscore(mesh_path, cloud, tau_frac=0.02, n=8000, seed=2):
    """The frozen v2 selector: F-score against the cloud at 2% of the CLOUD's extent."""
    m = trimesh.load(str(mesh_path), force="mesh")
    if len(getattr(m, "faces", [])) == 0:
        return -1.0
    tau = max(tau_frac * float(np.max(np.ptp(cloud, axis=0))), 1e-9)
    pts, _ = trimesh.sample.sample_surface(m, n, seed=seed)
    dp, _ = cKDTree(cloud).query(pts, k=1)
    do, _ = cKDTree(pts).query(cloud, k=1)
    pr, rc = float(np.mean(dp <= tau)), float(np.mean(do <= tau))
    return 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)


def bootstrap_ci(v, n=20000, seed=0):
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    a = np.asarray(v, float)
    m = a[rng.randint(0, len(a), (n, len(a)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    rows = json.loads((REPO / "updated_2/outputs/stage1/results.json").read_text())
    cells = defaultdict(dict)
    for r in rows:
        cells[(r["name"], r["mode"], r["level"], r["condition"])][int(r["seed"])] = r

    out = []
    for key, seeds in sorted(cells.items()):
        obj, mode, level, cond = key
        if len(seeds) < 2:
            continue
        cloud = cloud_seen_at_inference(obj, mode, level, cond)
        scored = {}
        for s, r in seeds.items():
            mp = REPO / r["mesh"]
            if mp.exists():
                scored[s] = (cloud_fscore(mp, cloud), r)
        if len(scored) < 2 or 0 not in scored:
            continue
        pick = max(scored, key=lambda s: scored[s][0])
        out.append(dict(obj=obj, mode=mode, level=level, cond=cond, pick=int(pick),
                        first_leak=float(scored[0][1]["leak_pct"]),
                        sel_leak=float(scored[pick][1]["leak_pct"]),
                        first_extra=float(scored[0][1]["extra_pct"]),
                        sel_extra=float(scored[pick][1]["extra_pct"]),
                        first_miss=float(scored[0][1]["missing_pct"]),
                        sel_miss=float(scored[pick][1]["missing_pct"])))
    json.dump(out, open(REPO / "updated_2/outputs/seedconsensus_vs_leak.json", "w"), indent=2)

    print("v2 seed consensus applied to CLUTTERED reconstructions (K=3)\n")
    print("positive dLeak = consensus left MORE of the intruder in the mesh\n")
    print(f"{'mode':8}{'n':>4}{'Leak 1draw':>12}{'Leak v2':>10}{'dLeak':>9}{'95% CI':>18}{'dExtra':>9}")
    clut = [r for r in out if r["cond"] == "clutter"]
    for mode in ("image", "points", "both", "ALL"):
        sub = clut if mode == "ALL" else [r for r in clut if r["mode"] == mode]
        if not sub:
            continue
        dl = [r["sel_leak"] - r["first_leak"] for r in sub]
        de = [r["sel_extra"] - r["first_extra"] for r in sub]
        lo, hi = bootstrap_ci(dl)
        mark = " *" if (hi < 0 or lo > 0) else ""
        print(f"{mode:8}{len(sub):>4}{np.mean([r['first_leak'] for r in sub]):>12.2f}"
              f"{np.mean([r['sel_leak'] for r in sub]):>10.2f}{np.mean(dl):>9.2f}"
              f"   [{lo:+.2f},{hi:+.2f}]{mark:>3}{np.mean(de):>9.2f}")

    print("\nsanity check — the same selector on CLEAN runs (no intruder exists)")
    cl = [r for r in out if r["cond"] == "clean"]
    if cl:
        de = [r["sel_extra"] - r["first_extra"] for r in cl]
        lo, hi = bootstrap_ci(de)
        print(f"  n={len(cl)}  dExtra {np.mean(de):+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")

    print("\nhow often consensus kept ShapeR's own draw:")
    for mode in ("image", "points", "both"):
        sub = [r for r in clut if r["mode"] == mode]
        if sub:
            print(f"  {mode:8} kept draw 0 on {sum(r['pick']==0 for r in sub)}/{len(sub)}")


if __name__ == "__main__":
    main()
