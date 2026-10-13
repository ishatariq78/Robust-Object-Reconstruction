# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Slide-sized version of the "pot in both photos and scan dots" figure.

The original sheet (`stage1/comparison/both_L040.png`) is correct — verified against
results.json directly — but it is FIVE rows tall, built for a report page, not a slide. Shrunk to
fit a 16:9 slide body, captions from adjacent rows visually collide with the row above, which
reads as a data problem when it is a formatting one.

This picks the three objects with the clearest effect at the 0.40 level and renders them large,
with clean/clutter defined explicitly in the caption:

    clean    baseline reconstruction — no pot in the photos, no pot in the scan dots
    clutter  pot painted into the photos AND added to the scan dots (mode=both)

    python updated_2/fig_both_clean.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "updated_2"))
import metrics as M                                        # noqa: E402

OUT = REPO / "updated_2/outputs/stage1"
INK, MUTED = (20, 22, 28), (108, 114, 124)
GT_C = np.array([104, 176, 132, 255], np.uint8)
INTR_C = np.array([214, 118, 96, 255], np.uint8)
OK_C = np.array([168, 175, 186, 255], np.uint8)
LEAK_C = np.array([201, 62, 42, 255], np.uint8)
OBJECTS = ["ADT1292__woodentable", "HWD0045__desktable", "ADT1292__coffeetable"]
LEVEL = 0.40


def font(sz, bold=False):
    p = f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"
    try:
        return ImageFont.truetype(p, sz)
    except Exception:
        return ImageFont.load_default()


def render(mesh, centre, radius, cols, size):
    import os
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    m = mesh.copy()
    m.visual = trimesh.visual.ColorVisuals(m, face_colors=cols)
    yfov = np.pi / 5.0
    e, a = np.radians(20.0), np.radians(35.0)
    d = 1.25 * radius / np.tan(yfov / 2)
    eye = centre + d * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    fwd = centre - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    pose = np.eye(4); pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, up, -fwd, eye
    sc = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.55, 0.55, 0.58])
    sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    sc.add(pyrender.PerspectiveCamera(yfov=yfov), pose=pose)
    sc.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=pose)
    r = pyrender.OffscreenRenderer(*size)
    try:
        rgb, _ = r.render(sc)
    finally:
        r.delete()
    return Image.fromarray(rgb.astype(np.uint8))


def leak_colors(mesh, target, distractor, thresh=M.THRESH):
    if distractor is None:
        return np.tile(OK_C, (len(mesh.faces), 1))
    fc = np.asarray(mesh.triangles_center, np.float64)
    ext = float(np.max(target.extents))
    tol = thresh * ext
    gp = M._sample(target, M.N_SAMPLES)
    dp = M._sample(distractor, M.N_SAMPLES)
    d_t, _ = cKDTree(gp).query(fc, k=1)
    d_d, _ = cKDTree(dp).query(fc, k=1)
    leaked = (d_d < d_t) & (d_d < tol)
    return np.where(leaked[:, None], LEAK_C[None, :], OK_C[None, :]).astype(np.uint8)


def main():
    s1 = json.loads((OUT / "results.json").read_text())
    by = {}
    for r in s1:
        if r["mode"] == "both" and r["level"] == LEVEL and r["seed"] == 0:
            by.setdefault(r["name"], {})[r["condition"]] = r

    cell, labw, hdr, gap = 300, 230, 96, 10
    cols = ["ground truth", "with the pot placed", "clean\n(no pot in either input)",
            "clutter\n(pot in photos + scan dots)"]
    W = labw + len(cols) * (cell + gap)
    H = hdr + len(OBJECTS) * (cell + 40)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((16, 14), "Pot in the photos AND the scan dots", fill=INK, font=font(28, True))
    d.text((16, 52), "red = predicted surface sitting on the pot rather than the table "
                     "(the same test Leak%% scores)", fill=MUTED, font=font(15))
    for i, c in enumerate(cols):
        x = labw + i * (cell + gap)
        for li, line in enumerate(c.split("\n")):
            tw = d.textlength(line, font=font(14, True))
            d.text((x + (cell - tw) / 2, hdr - 30 + li * 17), line, fill=MUTED, font=font(14, True))

    for r, name in enumerate(OBJECTS):
        rec = by.get(name)
        if not rec or "clean" not in rec or "clutter" not in rec:
            continue
        clean, clutter = rec["clean"], rec["clutter"]
        clut_pkl = REPO / f"updated_2/outputs/stage1/clutter/{name}__both__L040.pkl"
        lab = REPO / f"updated_2/outputs/stage1/clutter/{name}__both__L040__label.npz"
        z = np.load(lab)
        tgt = trimesh.Trimesh(vertices=z["target_vertices"], faces=z["target_faces"], process=False)
        intr = trimesh.Trimesh(vertices=z["distractor_vertices"], faces=z["distractor_faces"],
                               process=False)
        y = hdr + r * (cell + 40)
        allv = np.vstack([tgt.vertices, intr.vertices])
        centre = (allv.min(0) + allv.max(0)) / 2
        radius = float(np.linalg.norm(allv.max(0) - allv.min(0))) * 0.5

        d.text((16, y + cell // 2 - 8), name.split("__")[-1], fill=INK, font=font(18, True))
        gt_cols = np.tile(GT_C, (len(tgt.faces), 1))          # was OK_C (grey) — ground truth is green
        sheet.paste(render(tgt, centre, radius, gt_cols, (cell, cell)), (labw, y))
        ref = trimesh.util.concatenate([tgt, intr])
        ref_cols = np.concatenate([np.tile(GT_C, (len(tgt.faces), 1)),
                                   np.tile(INTR_C, (len(intr.faces), 1))])
        sheet.paste(render(ref, centre, radius, ref_cols, (cell, cell)), (labw + cell + gap, y))
        for i, rec_ in enumerate((clean, clutter)):
            p = Path(rec_["mesh"])
            if not p.exists():
                continue
            m = trimesh.load(str(p), force="mesh")
            dis = intr if rec_["condition"] == "clutter" else None
            cols_ = leak_colors(m, tgt, dis)
            x = labw + (i + 2) * (cell + gap)
            sheet.paste(render(m, centre, radius, cols_, (cell, cell)), (x, y))
            note = f"leak {rec_['leak_pct']:.1f}%  extra {rec_['extra_pct']:.1f}%  miss {rec_['missing_pct']:.1f}%"
            tw = d.textlength(note, font=font(13))
            d.text((x + (cell - tw) / 2, y + cell + 5), note, fill=MUTED, font=font(13))
    dst = OUT / "comparison" / "both_L040_slide.png"
    sheet.save(dst)
    print(f"wrote {dst}  ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
