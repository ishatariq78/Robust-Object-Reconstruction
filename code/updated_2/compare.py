# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Visual comparison sheets — one row per object.

The point of these sheets is to make the leakage visible, not just measurable, so
predicted surface that sits on the injected distractor is painted RED using exactly
the same test `metrics.py` scores as Leak%. A reader can then check the number
against the picture instead of taking it on trust.

ONE camera per object, framed on that object's ground-truth mesh at metric scale and
reused for every panel in the row. Framing each panel independently would make a
larger reconstruction look the same size as a smaller one and hide the very error
being shown.

    python updated_2/compare.py --results updated_2/outputs/stage1/results.json
    python updated_2/compare.py --results updated_2/outputs/stage2/results.json \
        --group method --out updated_2/outputs/stage2/comparison
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import metrics as M  # noqa: E402

BASE_RGBA = np.array([168, 175, 186, 255], np.uint8)
LEAK_RGBA = np.array([214, 69, 65, 255], np.uint8)
GT_RGBA = np.array([132, 168, 140, 255], np.uint8)


def font(sz, bold=False):
    p = f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"
    try:
        return ImageFont.truetype(p, sz)
    except Exception:
        return None


def leak_colors(mesh, target, distractor, thresh=M.THRESH):
    """
    Per-face colour: red where the face sits on the distractor rather than the
    target. Same rule as `metrics.score`, applied to face centres instead of
    surface samples so it can be painted.
    """
    if distractor is None or len(distractor.faces) == 0:
        return np.tile(BASE_RGBA, (len(mesh.faces), 1))
    fc = np.asarray(mesh.triangles_center, np.float64)
    ext = float(np.max(target.extents))
    tol = thresh * ext
    gp = M._sample(target, M.N_SAMPLES)
    dp = M._sample(distractor, M.N_SAMPLES)
    d_t, _ = cKDTree(gp).query(fc, k=1)
    d_d, _ = cKDTree(dp).query(fc, k=1)
    leaked = (d_d < d_t) & (d_d < tol)
    cols = np.tile(BASE_RGBA, (len(mesh.faces), 1))
    cols[leaked] = LEAK_RGBA
    return cols


def render(mesh, centre, radius, colors=None, size=(430, 430), elev=20.0, azim=35.0):
    """Z-up orbit; distance solved from the FOV so the whole object fits."""
    import pyrender

    m = mesh.copy()
    if len(m.faces) == 0:
        return Image.new("RGB", size, (247, 247, 250))
    if colors is None:
        colors = np.tile(BASE_RGBA, (len(m.faces), 1))
    m.visual = trimesh.visual.ColorVisuals(m, face_colors=colors)
    yfov = np.pi / 5.0
    e, a = np.radians(elev), np.radians(azim)
    d = 1.25 * radius / np.tan(yfov / 2)
    eye = centre + d * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    fwd = centre - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, up, -fwd, eye
    sc = pyrender.Scene(bg_color=[0.97, 0.97, 0.98, 1.0], ambient_light=[0.45, 0.45, 0.48])
    sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    sc.add(pyrender.PerspectiveCamera(yfov=yfov), pose=pose)
    sc.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.9), pose=pose)
    r = pyrender.OffscreenRenderer(*size)
    try:
        rgb, _ = r.render(sc)
    finally:
        r.delete()
    return Image.fromarray(rgb.astype(np.uint8))


def label(img, text, sub=None):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 26], fill=(255, 255, 255))
    d.text((8, 5), text, fill=(20, 20, 24), font=font(15, True))
    if sub:
        d.rectangle([0, img.height - 24, img.width, img.height], fill=(255, 255, 255))
        d.text((8, img.height - 20), sub, fill=(90, 90, 100), font=font(13))
    return img


def build_sheet(rows, group_key, out_path, tile=430):
    """
    One row per object; one column per value of `group_key`, plus GT and the
    GT+distractor reference.
    """
    by_obj = {}
    for r in rows:
        by_obj.setdefault(r["name"], []).append(r)

    cols_order, panels = None, []
    for obj, rs in sorted(by_obj.items()):
        label_npz = rs[0].get("label")
        if label_npz and Path(label_npz).exists():
            tgt, dis = M.load_label(label_npz)
        else:
            tgt, dis = None, None
        if tgt is None:
            continue
        centre = tgt.bounds.mean(axis=0)
        radius = float(np.linalg.norm(tgt.extents)) / 2.0
        if dis is not None:
            both = trimesh.util.concatenate([tgt, dis])
            centre = both.bounds.mean(axis=0)
            radius = float(np.linalg.norm(both.extents)) / 2.0

        # keep the lowest-seed row per group value, so the sheet is reproducible
        chosen = {}
        for r in rs:
            k = str(r.get(group_key))
            if k not in chosen or r.get("seed", 0) < chosen[k].get("seed", 0):
                chosen[k] = r
        keys = sorted(chosen)
        cols_order = cols_order or keys

        row_imgs = []
        gt_img = render(tgt, centre, radius, np.tile(GT_RGBA, (len(tgt.faces), 1)),
                        (tile, tile))
        row_imgs.append(label(gt_img, "ground truth", obj))
        if dis is not None:
            ref = trimesh.util.concatenate([tgt, dis])
            cols = np.concatenate([np.tile(GT_RGBA, (len(tgt.faces), 1)),
                                   np.tile(LEAK_RGBA, (len(dis.faces), 1))])
            row_imgs.append(label(render(ref, centre, radius, cols, (tile, tile)),
                                  "GT + intruder", "what was added"))
        for k in keys:
            r = chosen[k]
            p = r.get("mesh")
            if not p or not Path(p).exists():
                row_imgs.append(label(Image.new("RGB", (tile, tile), (247, 247, 250)),
                                      k, "missing"))
                continue
            mesh = trimesh.load(p, force="mesh")
            cols = leak_colors(mesh, tgt, dis if r.get("condition") != "clean" else None)
            sub = (f"leak {r.get('leak_pct', float('nan')):.1f}%  "
                   f"extra {r.get('extra_pct', float('nan')):.1f}%  "
                   f"miss {r.get('missing_pct', float('nan')):.1f}%")
            row_imgs.append(label(render(mesh, centre, radius, cols, (tile, tile)),
                                  k, sub))
        panels.append(row_imgs)

    if not panels:
        print("nothing to draw")
        return
    w = max(len(p) for p in panels) * tile
    h = len(panels) * tile
    sheet = Image.new("RGB", (w, h + 34), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((10, 8), f"red = predicted surface on the intruder (same test as Leak%)",
           fill=(40, 40, 48), font=font(17, True))
    for i, row in enumerate(panels):
        for j, im in enumerate(row):
            sheet.paste(im, (j * tile, 34 + i * tile))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"wrote {out_path}  ({sheet.width}x{sheet.height})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--group", default="condition",
                    help="column key: condition (stage 1) or method (stage 2)")
    ap.add_argument("--mode", default=None, help="filter stage-1 rows to one mode")
    ap.add_argument("--level", type=float, default=None)
    ap.add_argument("--clutter_dir", default=None,
                    help="where the __label.npz files live (default: beside the "
                         "results, falling back to stage1/clutter)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = json.loads(Path(args.results).read_text())
    if args.mode:
        rows = [r for r in rows if r.get("mode") == args.mode]
    if args.level is not None:
        rows = [r for r in rows if r.get("level") == args.level]

    # Attach the label path so the sheet can colour leakage.
    #
    # Stage 1 rows carry `mode`/`level` separately and their clutter sits beside the
    # results file. Stage 2 rows instead carry the full clutter tag in `name`
    # (e.g. `ADT1292__woodentable__points__L025`) and their clutter lives in the
    # STAGE 1 directory, so the default must point there rather than next to the
    # stage-2 results, where nothing would be found.
    clutter_dir = (Path(args.clutter_dir) if args.clutter_dir
                   else Path(args.results).parent / "clutter")
    if not clutter_dir.exists():
        alt = Path(args.results).parents[1] / "stage1" / "clutter"
        if alt.exists():
            clutter_dir = alt
    for r in rows:
        if "level" in r and "mode" in r:
            tag = f"{r['name']}__{r['mode']}__L{int(round(r['level']*100)):03d}__label.npz"
        else:
            tag = f"{r['name']}__label.npz"          # name already carries mode+level
        p = clutter_dir / tag
        if p.exists():
            r["label"] = str(p)

    out = args.out or str(Path(args.results).parent / "comparison" / "sheet.png")
    build_sheet(rows, args.group, out)


if __name__ == "__main__":
    main()
