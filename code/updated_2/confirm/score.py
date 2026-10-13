# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Score the abstention confirmation against the frozen protocol.

Imports every threshold and the verdict rule from `protocol.py` rather than restating them, so
this file cannot quietly disagree with what was registered.

The gate reads ONLY the baseline mesh and the SLAM cloud — never ground truth, never Leak%.
Ground truth enters afterwards, to report what each decision cost.

    python updated_2/confirm/score.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pickle
import trimesh
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "updated_2"))

import protocol as P                                   # noqa: E402
from abstain_probe import detector_scores              # noqa: E402


def cloud_recall(mesh_path, cloud, tau_frac=P.HEALTH_TAU_FRAC, n=8000, seed=2):
    """Share of OBSERVED points the mesh explains. Computable at inference; Missing% is not."""
    m = trimesh.load(str(mesh_path), force="mesh")
    if len(getattr(m, "faces", [])) == 0:
        return 0.0
    scale = float(np.max(np.ptp(cloud, axis=0)))
    pts, _ = trimesh.sample.sample_surface(m, n, seed=seed)
    d, _ = cKDTree(pts).query(cloud, k=1)
    return float(np.mean(d <= tau_frac * scale))


def bootstrap_ci(v, n=20000, seed=0):
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    a = np.asarray(v, float)
    m = a[rng.randint(0, len(a), (n, len(a)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def resolve(p):
    p = Path(p)
    if p.exists():
        return p
    g = sorted(p.parent.glob(p.name + "*"))
    return g[0] if g else None


def main():
    rows = []
    for res in sorted((HERE / "outputs").glob("stage_half*/results.json")):
        rows += json.loads(res.read_text())
    if not rows:
        raise SystemExit("no results yet — are the jobs still running?")

    by = {}
    for r in rows:
        by[(r["name"], str(r["seed"]), r["method"])] = r

    clouds, recs = {}, []
    for (cfg, sd, meth), r in sorted(by.items()):
        if meth != "baseline":
            continue
        rf = by.get((cfg, sd, "refeed"))
        if not rf:
            continue
        obj = "__".join(cfg.split("__")[:2])
        if obj not in clouds:
            d = pickle.load(open(REPO / f"data/{obj}.pkl", "rb"))
            clouds[obj] = d["points_model"].numpy().astype(np.float64)[:, :3]
        mp = resolve(REPO / r["mesh"]) or resolve(Path(r["mesh"]))
        if mp is None:
            print(f"  missing baseline mesh for {cfg} s{sd}")
            continue
        health = cloud_recall(mp, clouds[obj])
        leakscore = float(detector_scores(str(mp), clouds[obj])["top_unsup"])
        recs.append(dict(obj=obj, seed=sd, health=health, leakscore=leakscore,
                         act=bool(P.should_act(health, leakscore)),
                         b_leak=float(r["leak_pct"]), b_miss=float(r["missing_pct"]),
                         r_leak=float(rf["leak_pct"]), r_miss=float(rf["missing_pct"])))
    if not recs:
        raise SystemExit("no baseline/refeed pairs found")
    json.dump(recs, open(HERE / "outputs" / "confirm_scores.json", "w"), indent=2)

    per = defaultdict(list)
    for r in recs:
        per[r["obj"]].append(r)
    print(f"THE GATE'S DECISIONS  (threshold: cloud recall >= {P.HEALTH_THRESHOLD})\n")
    print(f"{'object':28}{'health':>8}{'decision':>10}{'Leak base':>11}{'Leak refeed':>13}"
          f"{'Miss base':>11}{'Miss refeed':>13}")
    for o in sorted(per, key=lambda x: -np.mean([r['health'] for r in per[x]])):
        g = per[o]
        act = np.mean([r["act"] for r in g]) > 0.5
        print(f"{o:28}{np.mean([r['health'] for r in g]):>8.3f}{'ACT' if act else 'abstain':>10}"
              f"{np.mean([r['b_leak'] for r in g]):>11.2f}{np.mean([r['r_leak'] for r in g]):>13.2f}"
              f"{np.mean([r['b_miss'] for r in g]):>11.2f}{np.mean([r['r_miss'] for r in g]):>13.2f}")

    acted = [r for r in recs if r["act"]]
    skipped = [r for r in recs if not r["act"]]
    print(f"\nacted on {len(acted)} runs / abstained on {len(skipped)} "
          f"({len({r['obj'] for r in acted})} vs {len({r['obj'] for r in skipped})} objects)")

    # PRIMARY: would always-acting have harmed what the gate skipped?
    dmg = [r["r_miss"] - r["b_miss"] for r in skipped]
    dlo, dhi = bootstrap_ci(dmg) if dmg else (float("nan"), float("nan"))
    print(f"\nPRIMARY — damage always-refeed WOULD have done to the abstained objects")
    print(f"  dMissing {np.mean(dmg) if dmg else float('nan'):+.2f}  95% CI [{dlo:+.2f}, {dhi:+.2f}]")
    print(f"  the gate's own damage there is 0.00 by construction (it does not act)")

    # SECONDARY: benefit where it did act, and share of total benefit retained
    dl = [r["r_leak"] - r["b_leak"] for r in acted]
    llo, lhi = bootstrap_ci(dl) if dl else (float("nan"), float("nan"))
    tot_all = sum(r["b_leak"] - r["r_leak"] for r in recs)
    tot_gate = sum(r["b_leak"] - r["r_leak"] for r in acted)
    retained = tot_gate / tot_all if abs(tot_all) > 1e-9 else float("nan")
    print(f"\nSECONDARY — benefit where the gate acted")
    print(f"  dLeak {np.mean(dl) if dl else float('nan'):+.2f}  95% CI [{llo:+.2f}, {lhi:+.2f}]")
    print(f"  benefit retained vs always-refeed: {retained*100:.0f}% "
          f"(need >= {P.MIN_BENEFIT_RETAINED*100:.0f}%)")

    print("\nPOLICY COMPARISON (all runs)")
    print(f"  {'policy':22}{'Leak%':>9}{'Missing%':>11}")
    print(f"  {'do nothing':22}{np.mean([r['b_leak'] for r in recs]):>9.2f}"
          f"{np.mean([r['b_miss'] for r in recs]):>11.2f}")
    print(f"  {'always refeed':22}{np.mean([r['r_leak'] for r in recs]):>9.2f}"
          f"{np.mean([r['r_miss'] for r in recs]):>11.2f}")
    print(f"  {'GATED':22}{np.mean([r['r_leak'] if r['act'] else r['b_leak'] for r in recs]):>9.2f}"
          f"{np.mean([r['r_miss'] if r['act'] else r['b_miss'] for r in recs]):>11.2f}")

    print(f"\nVERDICT: {P.verdict((dlo, dhi), (llo, lhi), retained)}")


if __name__ == "__main__":
    main()
