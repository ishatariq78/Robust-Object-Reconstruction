# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Stage 2 analysis, with the object-selection rule fixed IN ADVANCE.

Why this script exists
----------------------
Stage 1 showed the limitation is strongly object-dependent: of five tables, two leak
badly, one moderately, and two show nothing at all. On an object where the baseline
does not leak, a suppression method has nothing to remove and can only do harm, so
pooling all five dilutes any real effect toward zero and makes the comparison
uninformative in both directions.

The rule used here is therefore declared up front and depends ONLY on Stage 1:

    an object is "affected" if its mean ΔLeak (cluttered − clean, over seeds)
    in that clutter mode is at least `--min_delta_leak` points.

That is a property of the baseline model, measured before any method was run, so it
cannot be tuned to flatter a method. Both groups are always reported:

    AFFECTED   does the method remove leakage where there is leakage to remove?
    UNAFFECTED does the method leave already-clean objects alone? (harm check)

Reporting only the affected group would be cherry-picking; reporting only the pooled
number would hide the effect. Both, with the rule stated, is the honest form.

    python updated_2/analyze.py \
        --stage1 updated_2/outputs/stage1/results.json \
        --stage2 updated_2/outputs/stage2/results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402


def stage1_delta_leak(stage1_rows):
    """Mean ΔLeak per (object, mode, level), from the paired clean/clutter rows."""
    pairs = defaultdict(dict)
    for r in stage1_rows:
        pairs[(r["name"], r["mode"], r["level"], r["seed"])][r["condition"]] = r
    acc = defaultdict(list)
    for (name, mode, lv, _seed), v in pairs.items():
        if "clean" in v and "clutter" in v and "leak_pct" in v["clean"]:
            acc[(name, mode, lv)].append(v["clutter"]["leak_pct"] - v["clean"]["leak_pct"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def parse_stage2_name(name):
    """`ADT1292__woodentable__points__L025` -> (object, mode, level)."""
    m = re.match(r"^(.*)__(image|points|both)__L(\d+)$", name)
    if not m:
        return name, None, None
    return m.group(1), m.group(2), int(m.group(3)) / 100.0


def report(stage2_rows, dleak, min_delta_leak):
    groups = {"AFFECTED": [], "UNAFFECTED": []}
    for r in stage2_rows:
        obj, mode, lv = parse_stage2_name(r["name"])
        d = dleak.get((obj, mode, lv))
        if d is None:
            continue
        groups["AFFECTED" if d >= min_delta_leak else "UNAFFECTED"].append(
            {**r, "_obj": obj, "_mode": mode, "_level": lv, "_dleak": d})

    for gname, rows in groups.items():
        if not rows:
            print(f"\n[{gname}] no rows")
            continue
        objs = sorted({(r["_obj"], r["_mode"]) for r in rows})
        print(f"\n{'='*78}\n[{gname}]  {len(objs)} object-mode cells, "
              f"selection: Stage-1 ΔLeak "
              f"{'>=' if gname=='AFFECTED' else '<'} {min_delta_leak}")
        for o, md in objs:
            print(f"    {o} / {md}  (Stage-1 ΔLeak "
                  f"{dleak[(o, md, [r['_level'] for r in rows if r['_obj']==o and r['_mode']==md][0])]:+.1f})")

        methods = sorted({r["method"] for r in rows})
        print(f"\n{'method':10s} {'n':>3s} {'Leak%':>7s} {'Extra%':>8s} "
              f"{'Missing%':>9s} {'Chamfer%':>9s}")
        print("-" * 50)
        for m in methods:
            sel = [r for r in rows if r["method"] == m]
            a = M.aggregate(sel)
            print(f"{m:10s} {a['n']:3d} {a.get('leak_pct', np.nan):7.1f} "
                  f"{a.get('extra_pct', np.nan):8.1f} {a.get('missing_pct', np.nan):9.1f} "
                  f"{a.get('chamfer_pct', np.nan):9.2f}")

        base = {(r["_obj"], r["_mode"], r.get("seed")): r
                for r in rows if r["method"] == "baseline"}
        if not base:
            continue
        print(f"\n  paired vs baseline (negative = better for Leak/Extra; "
              f"Missing% rising = the method is eating the object)")
        for m in methods:
            if m == "baseline":
                continue
            sel = [r for r in rows if r["method"] == m]
            pr = [(base[k], r) for r in sel
                  if (k := (r["_obj"], r["_mode"], r.get("seed"))) in base]
            if not pr:
                continue
            line = f"  {m:8s} n={len(pr):2d} "
            for key, short in (("leak_pct", "Leak"), ("extra_pct", "Extra"),
                               ("missing_pct", "Miss")):
                x = [p[0].get(key) for p in pr]
                y = [p[1].get(key) for p in pr]
                if any(v is None for v in x + y):
                    continue
                st = M.paired_bootstrap(x, y)
                line += (f" | {short} {st['mean_diff']:+6.2f}"
                         f"[{st['ci_lo']:+.1f},{st['ci_hi']:+.1f}]"
                         f"{'*' if st['significant'] else ' '}")
            print(line)
    print("\n* = paired bootstrap 95% CI excludes zero (>=3 pairs required)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="updated_2/outputs/stage1/results.json")
    ap.add_argument("--stage2", default="updated_2/outputs/stage2/results.json")
    ap.add_argument("--min_delta_leak", type=float, default=2.0,
                    help="Stage-1 ΔLeak (points) above which an object counts as "
                         "actually exhibiting the limitation")
    args = ap.parse_args()

    s1 = json.loads(Path(args.stage1).read_text())
    dleak = stage1_delta_leak(s1)
    print("Stage-1 ΔLeak by (object, mode, level):")
    for k in sorted(dleak):
        print(f"  {k[0]:28s} {k[1]:7s} L{k[2]:.2f}  {dleak[k]:+6.1f}")

    p2 = Path(args.stage2)
    if not p2.exists():
        print(f"\n(no Stage 2 results yet at {p2})")
        return
    report(json.loads(p2.read_text()), dleak, args.min_delta_leak)


if __name__ == "__main__":
    main()
