# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Synthetic test data for ShapeR: occlusion, and a second object standing on the target.

Everything here modifies ONLY the image pixels (and, for occlusion, the point-projection mask
that follows from them). The SLAM point cloud, ground-truth mesh, cameras, crop, caption and
seed are never touched — so any difference in the reconstruction is caused by what the model
sees in the photos.

Two augmentations
  occlude(...)     paints a grey bar or blob over the object, hiding a measured fraction of its
                   silhouette, and drops the SLAM dots that fall behind it.
  put_on_target()  stands a real 3D mesh on the target's own flat surface and renders it through
                   EVERY view's own camera, so it is one object seen consistently from all views
                   (correct parallax), not a sticker pasted per image.

Camera conventions, verified rather than assumed (see `selftest_camera`):
  * `camera_extrinsics` is camera->world with the translation already scaled, i.e. it lives in
    the same frame as `semi_dense_points` and the dataset's scaled `vertices`.
  * `camera_intrinsics` is a pinhole K carried correctly through crop, resize and pad.
  * ShapeR/OpenCV cameras are +X right, +Y down, +Z forward; pyrender wants +Y up, -Z forward.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
MESH_CACHE = Path(__file__).resolve().parent / "distractor_meshes"

# A real object borrowed from another capture, used as the thing we stand on the target.
DISTRACTOR_SOURCES = {
    "plantpot": "data/BNB2909__plantpot.pkl",
    "bookstack": "data/HWD0045__stackedbooks.pkl",
    "mug": "data/HWD0045__coffeecup.pkl",
}


# =============================================================================
# camera
# =============================================================================


def sanitize_pose(c2w, tol=1e-4):
    """
    Clean affine 4x4 with bottom row exactly [0,0,0,1].

    The dataset builds camera_to_world with a float32 inverse, so the bottom row arrives as
    e.g. [4e-08, -2e-08, 5e-08, 0.99999988]. Our maths does not care; pyrender refuses it.
    We normalise and snap — but raise if the deviation is bigger than round-off, because then it
    is genuinely projective and flattening it would be wrong.
    """
    pose = np.asarray(c2w, dtype=np.float64).copy()
    if abs(pose[3, 3]) > 1e-12:
        pose = pose / pose[3, 3]
    dev = float(np.abs(pose[3] - np.array([0.0, 0.0, 0.0, 1.0])).max())
    if dev > tol:
        raise ValueError(f"camera_to_world is not affine (deviation {dev:.2e}); refusing to flatten")
    pose[3] = np.array([0.0, 0.0, 0.0, 1.0])
    return pose


def project_points(points, c2w, K):
    """points [N,3] in the scaled model frame -> (uv [N,2], depth [N])."""
    w2c = np.linalg.inv(sanitize_pose(c2w))
    cam = np.asarray(points, dtype=np.float64) @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * cam[:, 0] / safe + K[0, 2]
    v = K[1, 1] * cam[:, 1] / safe + K[1, 2]
    return np.stack([u, v], axis=1), z


def pose_opengl(c2w):
    """ShapeR/OpenCV camera -> pyrender/OpenGL camera."""
    pose = sanitize_pose(c2w)
    pose[:3, 1] *= -1.0
    pose[:3, 2] *= -1.0
    return pose


def selftest_camera(batch, max_median_px=5.0, verbose=True):
    """
    Project the sample's own 3D points and measure the median distance to ShapeR's own
    point-projection mask. Small error => our camera matches ShapeR's, so a mesh rendered
    through these cameras lands where we think it does.

    Distance, not a hit rate: the stored mask is sparse dots, so a correct projection landing
    3px from a dot centre still "misses" (that metric read 68% while the real error was 1.5px).
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(batch["semi_dense_points_orig"][0])[:, :3]
    ext = batch["camera_extrinsics"][0].cpu().numpy()
    itr = batch["camera_intrinsics"][0].cpu().numpy()
    masks = batch["masks_ingest"][0].float().cpu().numpy()
    if masks.ndim == 4:
        masks = masks[:, 0]
    v, h, w = masks.shape
    meds = []
    for i in range(v):
        uv, z = project_points(pts, ext[i], itr[i])
        ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        ys, xs = np.nonzero(masks[i] > 0.5)
        if ok.sum() == 0 or len(xs) == 0:
            continue
        d, _ = cKDTree(np.stack([xs, ys], 1).astype(float)).query(uv[ok], k=1)
        meds.append(float(np.median(d)))
    med = float(np.median(meds)) if meds else float("nan")
    passed = med <= max_median_px
    if verbose:
        print(f"  camera self-test: median reprojection error {med:.1f}px on a {w}px crop -> "
              f"{'PASS' if passed else 'FAIL'}")
    return {"median_px": med, "passed": bool(passed)}


# =============================================================================
# rendering meshes through the view cameras
# =============================================================================


def _renderer(h, w):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    return pyrender.OffscreenRenderer(w, h)


def render_through_views(mesh, extrinsics, intrinsics, h, w, renderer=None, intensity=1.6):
    """Render one mesh through every view camera -> (rgb [V,H,W,3], alpha [V,H,W], depth [V,H,W])."""
    import pyrender

    v = len(extrinsics)
    rgb = np.zeros((v, h, w, 3), np.float32)
    alpha = np.zeros((v, h, w), np.float32)
    depth = np.zeros((v, h, w), np.float32)
    own = renderer is None
    r = renderer or _renderer(h, w)
    try:
        for i in range(v):
            sc = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.15] * 3)
            mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.55, 0.55, 0.55, 1.0],
                                                     metallicFactor=0.0, roughnessFactor=0.85)
            sc.add(pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=False))
            K = intrinsics[i]
            cam = pyrender.IntrinsicsCamera(fx=float(K[0, 0]), fy=float(K[1, 1]),
                                            cx=float(K[0, 2]), cy=float(K[1, 2]),
                                            znear=0.005, zfar=100.0)
            p = pose_opengl(extrinsics[i])
            sc.add(cam, pose=p)
            sc.add(pyrender.DirectionalLight(color=np.ones(3), intensity=intensity), pose=p)
            col, dep = r.render(sc, flags=pyrender.RenderFlags.RGBA)
            rgb[i] = col[..., :3].astype(np.float32) / 255.0
            alpha[i] = (dep > 0).astype(np.float32)
            depth[i] = dep
    finally:
        if own:
            r.delete()
    return rgb, alpha, depth


def _paint_grey(img_chw, mask, rng, noise=0.02):
    """Fill `mask` with a grey drawn from THIS frame's own brightness, plus shading and noise."""
    h, w = mask.shape
    q = float(rng.uniform(0.08, 0.92))
    val = float(np.clip(np.quantile(img_chw[0], q), 0.12, 0.85))
    gy, gx = np.mgrid[0:h, 0:w]
    grad = ((gx / max(w - 1, 1) - 0.5) * rng.uniform(-0.12, 0.12)
            + (gy / max(h - 1, 1) - 0.5) * rng.uniform(-0.12, 0.12))
    patch = np.clip(val + grad + rng.normal(0, noise, (h, w)), 0, 1).astype(np.float32)
    out = img_chw.copy()
    for c in range(out.shape[0]):
        out[c] = out[c] * (1 - mask) + patch * mask
    return out


# =============================================================================
# augmentation 1 — occlusion
# =============================================================================


def _bar(fg, level, rng):
    h, w = fg.shape
    occ = np.zeros((h, w), np.float32)
    side = int(rng.integers(0, 4))
    total = float(fg.sum())
    if total < 1:
        return occ
    prof = fg.sum(axis=0) if side in (0, 1) else fg.sum(axis=1)
    if side in (1, 3):
        prof = prof[::-1]
    k = int(np.clip(np.searchsorted(np.cumsum(prof) / total, level) + 1, 0, len(prof)))
    if side == 0:   occ[:, :k] = 1
    elif side == 1: occ[:, w - k:] = 1
    elif side == 2: occ[:k, :] = 1
    else:           occ[h - k:, :] = 1
    return occ


def _blob(fg, level, rng):
    h, w = fg.shape
    total = float(fg.sum())
    if total < 1:
        return np.zeros((h, w), np.float32)
    ys, xs = np.nonzero(fg > 0.5)
    j = int(rng.integers(0, len(xs)))
    cx, cy = float(xs[j]), float(ys[j])
    asp, th = float(rng.uniform(0.6, 1.6)), float(rng.uniform(0, np.pi))
    r = float(np.sqrt(max(level, 1e-3) * total / np.pi))
    yy, xx = np.mgrid[0:h, 0:w]
    occ = np.zeros((h, w), np.float32)
    for _ in range(10):
        dx, dy = xx - cx, yy - cy
        u = (dx * np.cos(th) + dy * np.sin(th)) / max(r * asp, 1e-3)
        v = (-dx * np.sin(th) + dy * np.cos(th)) / max(r / asp, 1e-3)
        occ = ((u * u + v * v) <= 1.0).astype(np.float32)
        got = float((occ * fg).sum()) / total
        if abs(got - level) < 0.02 or got <= 0:
            break
        r *= float(np.clip(np.sqrt(level / max(got, 1e-3)), 0.6, 1.8))
    return occ


def occlude(images, point_masks, target_alpha, level, seed=0):
    """
    Hide `level` of the target's silhouette in every view.

    images       [V,3,H,W] in [0,1]
    point_masks  [V,1,H,W] ShapeR's point-projection mask
    target_alpha [V,H,W]   the target's silhouette (from its GT mesh, rendered through the views)

    Returns dict with the occluded images, the corrupted point mask, recomputed boxes, the
    occluder masks, and MEASURED statistics.
    """
    imgs = np.asarray(images, np.float32).copy()
    pm = np.asarray(point_masks, np.float32).copy()
    v, _c, h, w = imgs.shape
    rng = np.random.default_rng(seed)
    occ = np.zeros((v, h, w), np.float32)
    achieved, kept = [], []
    for i in range(v):
        fg = (target_alpha[i] > 0.5).astype(np.float32)
        if level <= 0 or fg.sum() < 1:
            achieved.append(0.0); kept.append(1.0); continue
        m = _bar(fg, level, rng) if rng.random() < 0.5 else _blob(fg, level, rng)
        imgs[i] = _paint_grey(imgs[i], m, rng)
        before = float((pm[i, 0] > 0.5).sum())
        pm[i, 0] = pm[i, 0] * (1 - m)
        occ[i] = m
        achieved.append(float((m * fg).sum()) / float(fg.sum()))
        kept.append(float((pm[i, 0] > 0.5).sum()) / before if before > 0 else 1.0)
    boxes = []
    for i in range(v):
        ys, xs = np.nonzero(pm[i, 0] > 0.5)
        boxes.append([0, 0, w, h] if len(xs) == 0 else
                     [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
    return {"images": imgs, "point_masks": pm,
            "boxes": np.asarray(boxes, np.float32).reshape(-1, 2, 2), "occluders": occ,
            "stats": {"requested": float(level), "achieved": float(np.mean(achieved)),
                      "points_kept": float(np.mean(kept))}}


# =============================================================================
# augmentation 2 — an object standing ON the target
# =============================================================================


def load_distractor(name="plantpot", force=False):
    """Normalised mesh (centred in XY, base at z=0, unit largest extent), cached as .ply."""
    MESH_CACHE.mkdir(parents=True, exist_ok=True)
    cache = MESH_CACHE / f"{name}.ply"
    if cache.exists() and not force:
        return trimesh.load(cache, force="mesh")
    d = pickle.load(open(REPO / DISTRACTOR_SOURCES[name], "rb"))
    m = trimesh.Trimesh(vertices=d["mesh_vertices"].numpy().astype(np.float64),
                        faces=d["mesh_faces"].numpy(), process=False)
    m.vertices -= np.array([m.bounds[:, 0][0] + m.extents[0] / 2,
                            m.bounds[:, 1][0] + m.extents[1] / 2, m.bounds[0, 2]])
    m.vertices /= max(float(np.max(m.extents)), 1e-6)
    m.export(cache)
    return m


def support_point(gt_mesh):
    """
    Where to stand something: the target's real support surface, from its upward-facing faces
    weighted by area (the sofa seat dominates that area; on a table, the top).

    Using the highest VERTEX instead puts the object on a sofa's armrest, where it floats beside
    the target and barely overlaps.
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


def put_on_target(images, gt_mesh, extrinsics, intrinsics, distractor="plantpot",
                  area_ratio=0.25, verbose=True):
    """
    Stand a real object on the target and composite it with a true depth test.

    The object is drawn where it is NEARER the camera than the target and hidden where the
    target is in front — a real occlusion, not a paste-over. Its size is solved for so its
    projected area is ~`area_ratio` of the target's; the ACHIEVED ratio is what gets reported.
    """
    imgs = np.asarray(images, np.float32).copy()
    v, _c, h, w = imgs.shape
    mesh = load_distractor(distractor)
    t, span, how = support_point(gt_mesh)

    r = _renderer(h, w)
    try:
        _rgb, t_alpha, t_depth = render_through_views(gt_mesh, extrinsics, intrinsics, h, w, r)

        def place(scale):
            m = mesh.copy()
            m.vertices = m.vertices * scale + t
            return render_through_views(m, extrinsics, intrinsics, h, w, r)

        def ratio_of(d_alpha):
            rs = [float((d_alpha[i] > .5).sum()) / float((t_alpha[i] > .5).sum())
                  for i in range(v) if (t_alpha[i] > .5).any()]
            return float(np.mean(rs)) if rs else 0.0

        lo, hi = 0.3 * span, 0.3 * span
        d_rgb, d_alpha, d_depth = place(lo)
        while ratio_of(d_alpha) < area_ratio and hi < 0.9 * span:
            lo, hi = hi, min(hi * 1.5, 0.9 * span)
            d_rgb, d_alpha, d_depth = place(hi)
        scale = hi
        if ratio_of(d_alpha) >= area_ratio:
            for _ in range(8):
                scale = 0.5 * (lo + hi)
                d_rgb, d_alpha, d_depth = place(scale)
                if ratio_of(d_alpha) < area_ratio:
                    lo = scale
                else:
                    hi = scale
    finally:
        r.delete()

    drawn = np.zeros((v, h, w), np.float32)
    rng = np.random.default_rng(0)
    for i in range(v):
        a = d_alpha[i] > 0.5
        if not a.any():
            continue
        td = np.where(t_depth[i] > 0, t_depth[i], np.inf)
        dd = np.where(d_depth[i] > 0, d_depth[i], np.inf)
        keep = a & (~(t_alpha[i] > 0.5) | (dd < td))     # in front of the target, or beside it
        if not keep.any():
            continue
        lum = d_rgb[i] @ np.array([0.299, 0.587, 0.114], np.float32)
        frame = imgs[i, 0]
        f_med = float(np.median(frame))
        f_sp = float(np.percentile(frame, 90) - np.percentile(frame, 10)) + 1e-6
        sel = lum[keep]
        s_med = float(np.median(sel))
        s_sp = float(np.percentile(sel, 90) - np.percentile(sel, 10)) + 1e-6
        lum = np.clip(float(np.clip(f_med * 0.82, .10, .85))
                      + (lum - s_med) * float(np.clip(1.25 * f_sp / s_sp, 0.8, 3.0)), 0, 1)
        for c in range(imgs.shape[1]):
            imgs[i, c][keep] = lum[keep]
        drawn[i] = keep.astype(np.float32)

    ratios = [float((drawn[i] > .5).sum()) / float((t_alpha[i] > .5).sum())
              for i in range(v) if (t_alpha[i] > .5).any()]
    stats = {"distractor": distractor, "placement": how, "requested_area": float(area_ratio),
             "achieved_area": float(np.mean(ratios)) if ratios else 0.0,
             "views_with_object": int(sum(1 for i in range(v) if drawn[i].any())),
             "n_views": int(v), "scale": float(scale)}
    if verbose:
        print(f"  {how}: object at {stats['achieved_area']*100:.0f}% of the target's area "
              f"(asked {area_ratio*100:.0f}%), visible in {stats['views_with_object']}/{v} views")
    return {"images": imgs, "drawn": drawn, "target_alpha": t_alpha, "stats": stats}
