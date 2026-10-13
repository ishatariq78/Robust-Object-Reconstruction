# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
The realistic case: a table with SEVERAL unknown things on it.

All the clutter data built so far has exactly ONE intruder, so nothing tested what happens with
a normal cluttered tabletop — which is the case that actually matters. This builds point clouds
with 1, 2 and 3 objects standing on the target and checks whether the segmenter finds ALL of
them without knowing what or how many they are.

Points only, no images: this tests the segmenter, which never sees an image.

Objects are placed on the target's own support surface at separated xy positions, scaled to a
realistic fraction of the target, and sampled at the target's own point density so no object is
easier to find than the target itself.

    python updated_2/multi_object_test.py
"""

import sys
from pathlib import Path

import numpy as np
import pickle
import trimesh

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "updated_shapeR/testing"))

from segment_v2 import segment_by_support_v2                      # noqa: E402
from latent_seg import segment_by_support                         # noqa: E402

TARGETS = ["ADT1292__woodentable", "HWD0045__desktable", "ADT1292__coffeetable",
           "HWD0045__woodentable", "BNB2909__coffeetable", "ADT1292__kitchenisland"]
PROPS = ["plantpot", "ADT1292__mug", "ADT1292__bowl", "ADT1292__jar"]


def load_prop(name):
    """
    Real object meshes as the things standing on the table.

    Only one prop mesh ships with the repo, so the rest are taken from the dataset's own ground
    truth — real scanned mugs, bowls and jars, not primitives. Using real shapes matters: a
    sphere would be far easier to separate than the awkward things people actually leave out.
    """
    ply = REPO / f"updated_shapeR/testing/distractor_meshes/{name}.ply"
    if ply.exists():
        return trimesh.load(str(ply), force="mesh")
    pkl = REPO / f"data/{name}.pkl"
    if pkl.exists():
        d = pickle.load(open(pkl, "rb"))
        return trimesh.Trimesh(vertices=d["mesh_vertices"].numpy().astype(np.float64),
                               faces=d["mesh_faces"].numpy(), process=False)
    return None


def support_height_and_top(cloud):
    """Where the tabletop is, from the cloud alone — the same information inference has."""
    z = cloud[:, 2]
    return float(np.percentile(z, 92))


def build(cloud, props, seed=0):
    """Stand `props` on the target's top surface at separated xy spots, same point density."""
    rng = np.random.RandomState(seed)
    z_top = support_height_and_top(cloud)
    near = cloud[cloud[:, 2] > np.percentile(cloud[:, 2], 80)]
    cx, cy = near[:, 0].mean(), near[:, 1].mean()
    span = float(np.linalg.norm(np.ptp(cloud[:, :2], axis=0)))
    density = len(cloud) / max(float(np.prod(np.ptp(cloud, axis=0))), 1e-9)

    pts, labels = [cloud], [np.ones(len(cloud), bool)]
    angles = np.linspace(0, 2 * np.pi, len(props), endpoint=False) + rng.rand()
    for prop, ang in zip(props, angles):
        m = prop.copy()
        target_h = 0.22 * span
        m.vertices *= target_h / max(float(np.max(m.extents)), 1e-9)
        r = 0.22 * span
        m.vertices += np.array([cx + r * np.cos(ang), cy + r * np.sin(ang),
                                z_top - float(m.vertices[:, 2].min())])
        npts = max(25, int(density * float(np.prod(m.extents)) * 0.5))
        s, _ = trimesh.sample.sample_surface(m, npts, seed=seed + 1)
        pts.append(np.asarray(s))
        labels.append(np.zeros(len(s), bool))
    return np.vstack(pts), np.concatenate(labels)


def main():
    props = [(n, load_prop(n)) for n in PROPS]
    props = [(n, m) for n, m in props if m is not None]
    if not props:
        raise SystemExit("no prop meshes found")
    print(f"props available: {[n for n, _ in props]}\n")
    print(f"{'target':26}{'objects on it':>15}{'OLD recall':>12}{'OLD fp':>9}"
          f"{'NEW recall':>12}{'NEW fp':>9}")
    agg = {}
    for t in TARGETS:
        p = REPO / f"data/{t}.pkl"
        if not p.exists():
            continue
        cloud = pickle.load(open(p, "rb"))["points_model"].numpy().astype(np.float64)[:, :3]
        for k in (1, 2, 3):
            pts, is_t = build(cloud, [m for _n, m in props[:k]], seed=k)
            row = []
            for fn in (segment_by_support, segment_by_support_v2):
                mask = fn(pts)
                rec = float((~mask & ~is_t).sum()) / max((~is_t).sum(), 1) * 100
                fp = float((~mask & is_t).sum()) / max(is_t.sum(), 1) * 100
                row += [rec, fp]
                agg.setdefault((fn.__name__, k), []).append((rec, fp))
            print(f"{t:26}{k:>15}{row[0]:>11.1f}%{row[1]:>8.2f}%{row[2]:>11.1f}%{row[3]:>8.2f}%")
    print()
    for k in (1, 2, 3):
        o = np.array(agg[("segment_by_support", k)])
        n = np.array(agg[("segment_by_support_v2", k)])
        print(f"{k} object(s) on the table:  OLD recall {o[:,0].mean():5.1f}% fp {o[:,1].mean():.2f}%"
              f"   |   NEW recall {n[:,0].mean():5.1f}% fp {n[:,1].mean():.2f}%")


if __name__ == "__main__":
    main()
