# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Build a CLUTTERED copy of a ShapeR sample: a second real object standing on the
target's own support surface, written into the pickle so that BOTH conditioning
branches see it.

Why this exists
---------------
The paper's Limitation (b) is: "when objects have other items stacked or closely
attached (for example, tables supporting other objects), the reconstructed meshes
sometimes include remnants of these adjacent structures".

An earlier bench in `updated_shapeR/testing/` painted a distractor into the IMAGES
only and left the SLAM point cloud untouched. Measured on the five held-out
objects, that produced no leakage at all (Extra% moved +0.4 / +0.4 / +1.4, and
twice went negative inside seed noise). ShapeR's geometry is carried by the
semi-dense point branch, so an image-only distractor is too weak a perturbation to
reproduce the failure the paper describes.

This module therefore injects the distractor into `points_model` as well, which is
the channel that actually drives the shape. Three modes let the two channels be
separated cleanly:

    mode=image    distractor in the pixels only          (reproduces the null result)
    mode=points   distractor in the SLAM points only     (isolates the point branch)
    mode=both     distractor in both                     (the realistic case)

What gets written
-----------------
Schema-preserving, index-aligned, and verified before it is saved:

  points_model              target points + sampled distractor points
  inv_dist_std, dist_std    kept aligned with points_model; added rows take the
                            MEDIAN of the originals (the one explicit imputation)
  visible_points_model      per view, distractor points that the view really sees
  object_point_projections  the matching projection rows, recomputed with the
                            repo's own Fisheye624 projection
  image_data                the view re-encoded with the distractor composited in
                            under a true depth test

The view pool is pruned first (see `--n_views`) so that the same pool is shared by
the clean and cluttered conditions; both then run ShapeR's own view selection over
identical candidates, so a difference between them is the clutter and nothing else.

Alongside the pkl it writes a `__label.npz` holding the PLACED distractor mesh.
That is the ground truth that makes leakage directly measurable: predicted surface
lying near the placed distractor is leaked geometry, which `metrics.py` reports as
Leak% rather than inferring it from Extra%.

    python updated_2/clutter.py --input_pkl data/ADT1292__woodentable.pkl \
        --level 0.25 --mode both --out_dir updated_2/outputs/clutter
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from experimental.reseg_shaper.geometry import (  # noqa: E402
    decode_view_image,
    project_points_fisheye624,
)

HERE = Path(__file__).resolve().parent
DISTRACTORS = HERE / "distractors"

# The RGB stream is the one `InferenceDataset` reads for these samples.
STREAM_KEYS = {
    "slam": dict(images="image_data", cameras="Ts_camera_model",
                 params="camera_params", visible="visible_points_model",
                 projections="object_point_projections"),
    "rgb": dict(images="rgb_image_data", cameras="Ts_rgbCamera_model",
                params="rgb_camera_params", visible="rgb_visible_points_model",
                projections="rgb_object_point_projections"),
}


def _key(p):
    """Identity key for a 3D point. float64 BEFORE rounding — rounding a float32
    and a float64 copy of the same value yields different keys."""
    return tuple(np.round(np.asarray(p, dtype=np.float64), 6))


# --------------------------------------------------------------------------- #
# placement
# --------------------------------------------------------------------------- #

def support_point(gt_mesh):
    """
    Where to stand something: the target's real support surface, taken from its
    upward-facing faces weighted by area (on a table, the top; on a sofa, the seat).

    Using the highest VERTEX instead puts the object on an armrest, where it floats
    beside the target and barely overlaps it.
    """
    v = np.asarray(gt_mesh.vertices, np.float64)
    span = float(np.max(np.ptp(v, axis=0)))
    fn = np.asarray(gt_mesh.face_normals, np.float64)
    fc = np.asarray(gt_mesh.triangles_center, np.float64)
    fa = np.asarray(gt_mesh.area_faces, np.float64)
    up = (fn[:, 2] > 0.85) & (fc[:, 2] > np.percentile(v[:, 2], 40))
    if up.sum() >= 10:
        aw = fa[up] / fa[up].sum()
        cx, cy = float((fc[up, 0] * aw).sum()), float((fc[up, 1] * aw).sum())
        near = fc[up][np.linalg.norm(fc[up][:, :2] - [cx, cy], axis=1) < 0.15 * span]
        z = float(np.percentile(near[:, 2], 90)) if len(near) else float(fc[up][:, 2].max())
        how = f"support surface ({int(up.sum())} up-facing faces)"
    else:
        top = v[v[:, 2] >= np.percentile(v[:, 2], 80)]
        cx, cy = float(np.median(top[:, 0])), float(np.median(top[:, 1]))
        near = v[np.linalg.norm(v[:, :2] - [cx, cy], axis=1) < 0.18 * span]
        z = float(near[:, 2].max()) if len(near) else float(v[:, 2].max())
        how = "no flat surface — highest region"
    return np.array([cx, cy, z]), span, how


def load_distractor(name="plantpot"):
    p = DISTRACTORS / f"{name}.ply"
    if not p.exists():
        raise SystemExit(f"no distractor mesh at {p}")
    m = trimesh.load(p, force="mesh")
    m.vertices = np.asarray(m.vertices, np.float64)
    # normalise: unit height, resting on z=0, centred in xy
    m.vertices -= m.vertices.mean(axis=0)
    m.vertices /= float(np.max(np.ptp(m.vertices, axis=0)))
    m.vertices[:, 2] -= m.vertices[:, 2].min()
    return m


def place_distractor(gt_mesh, name, level):
    """
    Stand the distractor on the target's support surface at `level` x the target's
    span. Returns the placed mesh and a description of the placement.
    """
    t, span, how = support_point(gt_mesh)
    m = load_distractor(name)
    m.vertices = m.vertices * (level * span) + t
    stats = {
        "distractor": name,
        "placement": how,
        "level": float(level),
        "target_span": span,
        "distractor_height": float(np.ptp(np.asarray(m.vertices)[:, 2])),
        "base_xyz": [float(x) for x in t],
    }
    return m, stats


# --------------------------------------------------------------------------- #
# point-cloud injection
# --------------------------------------------------------------------------- #

def sample_distractor_points(placed, target_points, seed=0, density_match=True):
    """
    Sample SLAM-like points on the distractor.

    The count is matched to the TARGET's own point density (points per unit surface
    area) rather than fixed, so the injected cloud is as sparse as the data ShapeR
    actually receives. A denser-than-real injection would overstate the effect.
    """
    rng = np.random.default_rng(seed)
    if density_match:
        # target surface area proxy: convex-hull area of its point cloud
        try:
            hull_area = float(trimesh.PointCloud(target_points).convex_hull.area)
        except Exception:
            hull_area = float(np.prod(np.ptp(target_points, axis=0)[:2])) + 1e-6
        density = len(target_points) / max(hull_area, 1e-6)
        n = int(np.clip(round(density * float(placed.area)), 12, 4 * len(target_points)))
    else:
        n = 300
    pts, _ = trimesh.sample.sample_surface(placed, n)
    # SLAM points are noisy; jitter by a fraction of the object's own size
    jitter = 0.004 * float(np.max(np.ptp(np.asarray(placed.vertices), axis=0)))
    pts = np.asarray(pts, np.float64) + rng.normal(0, jitter, size=(n, 3))
    return pts.astype(np.float32), n


class _Occluder:
    """
    Exact ray-cast visibility against the target mesh.

    open3d's RaycastingScene is used rather than trimesh's, because the trimesh
    path needs `rtree`, which is not installed in the `shaper` env. This is an
    exact geometric test either way — the pinhole render in `composite_images` is
    an approximation, but the point injection never relies on it.
    """

    def __init__(self, mesh):
        import open3d as o3d

        self._o3d = o3d
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(
                o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(np.asarray(mesh.vertices, np.float64)),
                    o3d.utility.Vector3iVector(np.asarray(mesh.faces, np.int32)))))

    def visible_from(self, eye, points):
        """True where the segment eye->point is not blocked by the target."""
        if len(points) == 0:
            return np.zeros(0, bool)
        d = np.asarray(points, np.float64) - np.asarray(eye, np.float64)[None, :]
        L = np.linalg.norm(d, axis=1)
        dirs = d / np.maximum(L, 1e-12)[:, None]
        origins = np.repeat(np.asarray(eye, np.float64)[None, :], len(points), 0)
        rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
        hit_t = self.scene.cast_rays(
            self._o3d.core.Tensor(rays))["t_hit"].numpy()
        # blocked only if the target is hit strictly BEFORE the point itself
        return ~(hit_t < (L - 1e-3))


def inject_points(sample, keys, dist_points, placed, target_mesh, verbose=True):
    """
    Append the distractor points to `points_model` and to every view that really
    observes them, keeping all index-aligned arrays consistent.

    A view "observes" a distractor point when the repo's own Fisheye624 projection
    says it is in front of the camera and inside the image, AND the point is not
    hidden behind the target (checked against the target mesh by ray casting).
    """
    out = dict(sample)

    points = sample["points_model"].numpy()
    inv_dist = sample["inv_dist_std"].numpy()
    dist = sample["dist_std"].numpy()

    med_inv = float(np.median(inv_dist)) if inv_dist.size else 0.0
    med_dist = float(np.median(dist)) if dist.size else 0.0

    out["points_model"] = torch.from_numpy(
        np.concatenate([points, dist_points], axis=0).astype(np.float32))
    out["inv_dist_std"] = torch.from_numpy(
        np.concatenate([inv_dist, np.full(len(dist_points), med_inv, np.float32)]))
    out["dist_std"] = torch.from_numpy(
        np.concatenate([dist, np.full(len(dist_points), med_dist, np.float32)]))

    # Occlude against target AND the distractor itself: a real SLAM point on the
    # far side of the pot is not observed from this view either. Testing only
    # against the target would inject a fully see-through object.
    occl = _Occluder(trimesh.util.concatenate([target_mesh, placed]))

    n_views = len(sample[keys["cameras"]])
    new_visible, new_projections = [], []
    n_added_obs = 0
    for vi in range(n_views):
        vis = sample[keys["visible"]][vi].numpy()
        proj = sample[keys["projections"]][vi].numpy()
        cam = sample[keys["cameras"]][vi].numpy()
        par = sample[keys["params"]][vi].numpy()
        dims = decode_view_image(sample, vi, keys["images"]).shape[:2]

        uv, _, valid = project_points_fisheye624(dist_points, cam, par, dims)
        if valid.any():
            # occlusion test: is the target between this point and the camera?
            c2w = np.linalg.inv(cam)
            eye = c2w[:3, 3]
            cand = dist_points[valid]
            seen = occl.visible_from(eye, cand)
        else:
            cand = np.zeros((0, 3), np.float32)
            seen = np.zeros(0, bool)

        add_pts = cand[seen] if len(cand) else np.zeros((0, 3), np.float32)
        add_uv = uv[valid][seen] if valid.any() else np.zeros((0, 2), np.float32)
        n_added_obs += len(add_pts)

        new_visible.append(torch.from_numpy(
            np.concatenate([vis, add_pts.astype(vis.dtype)], axis=0)))
        new_projections.append(torch.from_numpy(
            np.concatenate([proj, add_uv.astype(proj.dtype)], axis=0)))

    out[keys["visible"]] = new_visible
    out[keys["projections"]] = new_projections

    report = {
        "n_points_before": int(len(points)),
        "n_points_after": int(len(points) + len(dist_points)),
        "n_distractor_points": int(len(dist_points)),
        "distractor_point_fraction": float(len(dist_points) / (len(points) + len(dist_points))),
        "n_view_observations_added": int(n_added_obs),
        "mean_observations_per_view": float(n_added_obs / max(n_views, 1)),
        "imputed_inv_dist_std": med_inv,
        "imputed_dist_std": med_dist,
    }
    if verbose:
        print(f"  points {report['n_points_before']} -> {report['n_points_after']} "
              f"({report['distractor_point_fraction']*100:.1f}% distractor), "
              f"{report['mean_observations_per_view']:.1f} obs/view")
    return out, report


# --------------------------------------------------------------------------- #
# image compositing
# --------------------------------------------------------------------------- #

def _sanitize_pose(m, tol=1e-4):
    """Force an exact [0,0,0,1] bottom row and an orthonormal rotation.

    Stored poses drift by float error and pyrender rejects them outright.
    """
    m = np.asarray(m, np.float64).copy()
    assert np.allclose(m[3], [0, 0, 0, 1], atol=tol), f"pose bottom row too far off: {m[3]}"
    m[3] = [0, 0, 0, 1]
    u, _s, vt = np.linalg.svd(m[:3, :3])
    m[:3, :3] = u @ vt
    return m


_RENDERERS = {}


def _renderer(h, w):
    import pyrender

    if (h, w) not in _RENDERERS:
        _RENDERERS[(h, w)] = pyrender.OffscreenRenderer(w, h)
    return _RENDERERS[(h, w)]


def _render(mesh, cam, par, dims):
    """Render a mesh through one stored camera. Returns (luminance, alpha, depth)."""
    import pyrender

    h, w = dims
    # The stored cameras are Fisheye624; pyrender is pinhole. Rendering with the
    # pinhole part of the same intrinsics is accurate near the centre and drifts at
    # the edge, so the composite is depth-tested against the SAME approximation and
    # the point injection above never uses it.
    fx, fy, cx, cy = float(par[0]), float(par[0]), float(par[1]), float(par[2])
    # Shaded, not FLAT: an unshaded render has zero luminance variance, and the
    # contrast match below then collapses the object into a uniform blob that reads
    # as an occluder rather than as a second object.
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.35, 0.35, 0.35])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    c2w = _sanitize_pose(np.linalg.inv(_sanitize_pose(cam)))
    gl = _sanitize_pose(c2w @ np.diag([1.0, -1.0, -1.0, 1.0]))
    scene.add(pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0),
              pose=gl)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=gl)
    color, depth = _renderer(h, w).render(scene)
    lum = (np.asarray(color, np.float32)[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32))
    return lum, (depth > 0), depth


def composite_images(sample, keys, placed, target_mesh, verbose=True):
    """
    Paint the distractor into every view of the pool under a true depth test: it is
    drawn where it is NEARER the camera than the target, and hidden where the target
    is in front. Grey level is matched to each frame's own statistics so the paste
    does not read as a brightness anomaly.
    """
    out = dict(sample)
    n_views = len(sample[keys["cameras"]])
    blobs, n_painted = [], 0
    for vi in range(n_views):
        img = decode_view_image(sample, vi, keys["images"])
        dims = img.shape[:2]
        cam = sample[keys["cameras"]][vi].numpy()
        par = sample[keys["params"]][vi].numpy()
        d_lum, d_a, d_d = _render(placed, cam, par, dims)
        if not d_a.any():
            blobs.append(sample[keys["images"]][vi])
            continue
        _t_lum, t_a, t_d = _render(target_mesh, cam, par, dims)
        td = np.where(t_a, t_d, np.inf)
        dd = np.where(d_a, d_d, np.inf)
        keep = d_a & (~t_a | (dd < td))
        if not keep.any():
            blobs.append(sample[keys["images"]][vi])
            continue

        arr = np.asarray(img, np.float32)
        gray = arr if arr.ndim == 2 else arr.mean(axis=2)
        f_med = float(np.median(gray))
        f_sp = float(np.percentile(gray, 90) - np.percentile(gray, 10)) + 1e-6
        sel = d_lum[keep]
        s_med, s_sp = float(np.median(sel)), float(sel.std()) + 1e-6
        painted = np.clip((sel - s_med) / s_sp * (0.25 * f_sp) + f_med, 0, 255)

        newarr = arr.copy()
        if newarr.ndim == 2:
            newarr[keep] = painted
        else:
            newarr[keep] = painted[:, None]
        n_painted += 1

        buf = io.BytesIO()
        Image.fromarray(newarr.astype(np.uint8)).save(buf, format="PNG")
        blobs.append(buf.getvalue())

    out[keys["images"]] = blobs
    if verbose:
        print(f"  composited into {n_painted}/{n_views} views")
    return out, {"n_views_painted": int(n_painted), "n_views": int(n_views)}


# --------------------------------------------------------------------------- #
# view-pool pruning
# --------------------------------------------------------------------------- #

def prune_views(sample, keys, n_views):
    """
    Keep the n_views views that observe the target best (most stored projections).
    Both the clean and the cluttered condition are built from this same pool, so
    ShapeR's own view selection then runs over identical candidates.
    """
    counts = np.array([len(p) for p in sample[keys["projections"]]])
    chosen = sorted(np.argsort(-counts)[:n_views].tolist())
    out = dict(sample)
    out[keys["images"]] = [sample[keys["images"]][i] for i in chosen]
    out[keys["visible"]] = [sample[keys["visible"]][i] for i in chosen]
    out[keys["projections"]] = [sample[keys["projections"]][i] for i in chosen]
    out[keys["cameras"]] = sample[keys["cameras"]][chosen]
    out[keys["params"]] = sample[keys["params"]][chosen]
    return out, chosen


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #

def verify(sample, keys, original_n_points):
    """Fail loudly rather than silently producing an invalid sample."""
    p = sample["points_model"]
    assert p.ndim == 2 and p.shape[1] == 3, "points_model malformed"
    assert len(sample["inv_dist_std"]) == len(p), "inv_dist_std not aligned"
    assert len(sample["dist_std"]) == len(p), "dist_std not aligned"
    assert len(p) >= original_n_points, "points were lost"
    n = len(sample[keys["cameras"]])
    for k in ("images", "visible", "projections"):
        assert len(sample[keys[k]]) == n, f"{k} not aligned with cameras ({len(sample[keys[k]])} vs {n})"
    for vi in range(n):
        assert len(sample[keys["projections"]][vi]) > 0, (
            f"view {vi} has an empty projection list — ShapeR would collapse its crop box")
    return True


# --------------------------------------------------------------------------- #

def build(input_pkl, out_dir, level=0.25, mode="both", distractor="plantpot",
          n_views=48, stream="slam", seed=0, verbose=True):
    keys = STREAM_KEYS[stream]
    name = Path(input_pkl).stem
    with open(input_pkl, "rb") as f:
        sample = pickle.load(f)

    gt = trimesh.Trimesh(vertices=np.asarray(sample["mesh_vertices"], np.float64),
                         faces=np.asarray(sample["mesh_faces"]), process=False)
    placed, place_stats = place_distractor(gt, distractor, level)
    if verbose:
        print(f"[{name}] level={level} mode={mode} — {place_stats['placement']}")

    sample, chosen = prune_views(sample, keys, n_views)
    n0 = len(sample["points_model"])

    report = {"name": name, "mode": mode, "stream": stream,
              "n_views_pool": len(chosen), **place_stats}

    # clean condition: the same pruned pool, nothing added
    clean = dict(sample)

    cluttered = dict(sample)
    if mode in ("points", "both"):
        dpts, _n = sample_distractor_points(
            placed, sample["points_model"].numpy(), seed=seed)
        cluttered, rep = inject_points(cluttered, keys, dpts, placed, gt, verbose)
        report.update(rep)
    if mode in ("image", "both"):
        cluttered, rep = composite_images(cluttered, keys, placed, gt, verbose)
        report.update(rep)

    verify(clean, keys, n0)
    verify(cluttered, keys, n0)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{name}__{mode}__L{int(round(level*100)):03d}"

    clean_path = out_dir / f"{name}__clean.pkl"
    if not clean_path.exists():
        with open(clean_path, "wb") as f:
            pickle.dump(clean, f)

    clutter_path = out_dir / f"{tag}.pkl"
    with open(clutter_path, "wb") as f:
        pickle.dump(cluttered, f)

    # Ground truth for Leak%: the placed distractor, in the model frame.
    #
    # `n_target_points` records how many rows of the cluttered `points_model` are
    # the ORIGINAL target's — injected distractor points are appended after them.
    # That gives an exact oracle for which conditioning points belong to the
    # intruder, which the segmentation methods are scored against.
    n_target_points = int(len(sample["points_model"]))
    np.savez(out_dir / f"{tag}__label.npz",
             distractor_vertices=np.asarray(placed.vertices, np.float32),
             distractor_faces=np.asarray(placed.faces, np.int64),
             target_vertices=np.asarray(gt.vertices, np.float32),
             target_faces=np.asarray(gt.faces, np.int64),
             n_target_points=n_target_points,
             n_total_points=int(len(cluttered["points_model"])))
    (out_dir / f"{tag}__report.json").write_text(json.dumps(report, indent=2))

    if verbose:
        print(f"  wrote {clutter_path.name}  (clean: {clean_path.name})")
    return {"clean_pkl": str(clean_path), "clutter_pkl": str(clutter_path),
            "label": str(out_dir / f"{tag}__label.npz"), "report": report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_pkl", nargs="+", required=True)
    ap.add_argument("--out_dir", default="updated_2/outputs/clutter")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.25])
    ap.add_argument("--modes", nargs="+", default=["both"],
                    choices=["image", "points", "both"])
    ap.add_argument("--distractor", default="plantpot")
    ap.add_argument("--n_views", type=int, default=48)
    ap.add_argument("--stream", default="slam", choices=list(STREAM_KEYS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    made = []
    for pkl in args.input_pkl:
        for mode in args.modes:
            for lv in args.levels:
                made.append(build(pkl, args.out_dir, lv, mode, args.distractor,
                                  args.n_views, args.stream, args.seed))
    print(f"\nBuilt {len(made)} cluttered samples in {args.out_dir}")


if __name__ == "__main__":
    main()
