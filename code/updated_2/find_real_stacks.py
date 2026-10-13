# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Find REAL table-with-objects-on-it cases already present in the dataset.

Everything up to now injected a synthetic pot. That was never necessary. Each pickle is one
annotated object from a scene, `points_model` is metric and zero-centred, and `T_model_world`
places it back in the scene — so every object in a sequence can be put into ONE world frame and
the real spatial relationships read off directly.

That gives what synthetic clutter could only approximate: a real table, real objects genuinely
standing on it, and a real ground-truth mesh for each. Leakage can then be measured against the
neighbour's OWN annotation rather than something we placed ourselves.

`rescale_back` shows the convention: model -> world is inverse(T_model_world).

A pair (A, B) counts as "B stands on A" when B's underside sits near A's top surface and B's
footprint falls inside A's. Both thresholds are geometric and generous; the point is to surface
candidates for inspection, not to be the final word on what is stacked.

    python updated_2/find_real_stacks.py
"""

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def world_geom(pkl):
    """Object's ground-truth vertices in WORLD coordinates, plus its own local up in world."""
    d = pickle.load(open(pkl, "rb"))
    if "mesh_vertices" not in d or "T_model_world" not in d:
        return None
    v = d["mesh_vertices"].numpy().astype(np.float64)
    T = np.linalg.inv(d["T_model_world"].numpy().astype(np.float64))
    vw = v @ T[:3, :3].T + T[:3, 3]
    up = T[:3, :3] @ np.array([0.0, 0.0, 1.0])       # points_model is z-up per object
    return dict(v=vw, up=up / np.linalg.norm(up), n_pts=len(d["points_model"]),
                cap=str(d.get("caption", ""))[:50])


def main():
    """
    World-up is per SCENE, not global: ADT1292 resolves to +Y and BNB1045 to +Z, each with
    perfect internal agreement. An earlier version guessed one axis from object extents, picked
    X, and "found" a bedside lamp standing on a bed - objects that are beside each other. The up
    direction is taken from the objects own frames instead of guessed.
    """
    scenes = defaultdict(dict)
    for p in sorted((REPO / "data").glob("*.pkl")):
        if "__" not in p.stem:
            continue
        try:
            g = world_geom(p)
        except Exception:
            continue
        if g:
            scenes[p.stem.split("__")[0]][p.stem] = g
    print(f"{sum(len(v) for v in scenes.values())} objects across {len(scenes)} scenes\n")

    found = []
    for scene, objs in scenes.items():
        ups = np.array([g["up"] for g in objs.values()])
        up = ups.mean(0); up /= np.linalg.norm(up)
        if float(np.mean(ups @ up)) < 0.95:
            print(f"  {scene}: objects disagree on up - skipped")
            continue
        a = np.array([1.0, 0, 0])
        if abs(a @ up) > 0.9:
            a = np.array([0, 1.0, 0])
        e1 = np.cross(up, a); e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        box = {}
        for n, g in objs.items():
            loc = np.stack([g["v"] @ e1, g["v"] @ e2, g["v"] @ up], axis=1)
            box[n] = (loc.min(0), loc.max(0), g)
        for na, (alo, ahi, ga) in box.items():
            ha = ahi[2] - alo[2]
            for nb, (blo, bhi, gb) in box.items():
                if na == nb:
                    continue
                if np.prod(bhi - blo) > np.prod(ahi - alo):
                    continue
                gap = blo[2] - ahi[2]
                if not (-0.12 * max(ha, 1e-6) <= gap <= 0.10 * max(ha, 1e-6)):
                    continue
                ov = 1.0
                for i in (0, 1):
                    lo = max(alo[i], blo[i]); hi = min(ahi[i], bhi[i])
                    ov *= max(0.0, hi - lo) / max(bhi[i] - blo[i], 1e-9)
                if ov < 0.5:
                    continue
                found.append((na, nb, ov, ga["n_pts"], gb["n_pts"], gb["cap"]))

    found.sort(key=lambda r: (r[0], -r[2]))
    print(f"{'target (the thing you want)':32}{'standing on it':28}{'overlap':>9}"
          f"{'tgt pts':>9}{'obj pts':>9}")
    for na, nb, ov, pa, pb, cap in found:
        print(f"{na:32}{nb:28}{ov*100:>8.0f}%{pa:>9}{pb:>9}")
    print(f"\n{len(found)} real stacked pairs")
    tables = defaultdict(list)
    for na, nb, ov, pa, pb, cap in found:
        tables[na].append(nb)
    multi = {k: v for k, v in tables.items() if len(v) > 1}
    print(f"{len(tables)} distinct targets; {len(multi)} carry MORE THAN ONE object:")
    for k, v in sorted(multi.items(), key=lambda x: -len(x[1])):
        print(f"  {k:30} <- {', '.join(x.split('__')[-1] for x in v)}")
    import json
    json.dump([{"target": na, "on_it": nb, "overlap": ov} for na, nb, ov, _a, _b, _c in found],
              open(REPO / "updated_2/real_stacks.json", "w"), indent=2)


if __name__ == "__main__":
    main()
