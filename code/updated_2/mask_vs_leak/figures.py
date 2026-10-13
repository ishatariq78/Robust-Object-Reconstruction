# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Visuals for the mask fix: what the model was told, and what it built.

TWO FIGURES, because the change and its consequence are different claims.

  fig_masks   the input view, the mask the pipeline builds today, and the mask after the fix.
              Red overlay = pixels the model is told are the target. If the fix works, the
              intruder stops being red. This shows the CHANGE.

  fig_recon   ground truth (with the intruder marked), then the three reconstructions.
              Predicted surface is coloured RED where it sits on the intruder — the same test
              Leak% counts — and grey elsewhere. This shows the CONSEQUENCE.

One camera per object, framed on the ground truth at metric scale and reused for every mesh, so
a bloated reconstruction cannot look the same size as a good one.

    python updated_2/mask_vs_leak/figures.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "outputs" / "three_arms"

INK, MUTED, GREY = (20, 22, 28), (108, 114, 124), (150, 156, 166)
GOOD = np.array([170, 176, 186, 255], np.uint8)      # surface that is not on the intruder
LEAK = np.array([201, 62, 42, 255], np.uint8)        # surface sitting on the intruder
GT_C = np.array([104, 176, 132, 255], np.uint8)
INTR = np.array([214, 118, 96, 255], np.uint8)
ARMS = [("control", "guidance OFF"), ("stock", "mask as built today"), ("fixed", "mask fixed")]


def font(sz, bold=False):
    p = f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"
    try:
        return ImageFont.truetype(p, sz)
    except Exception:
        return ImageFont.load_default()


def clutter_pkl(name):
    for d in ("updated_2/outputs/stage1/clutter", "updated_2/confirm/outputs/clutter"):
        p = REPO / d / f"{name}.pkl"
        if p.exists():
            return p
    return None


def intruder_mesh(name):
    for d in ("updated_2/outputs/stage1/clutter", "updated_2/confirm/outputs/clutter"):
        p = REPO / d / f"{name}__label.npz"
        if p.exists():
            z = np.load(p)
            return trimesh.Trimesh(vertices=z["distractor_vertices"],
                                   faces=z["distractor_faces"], process=False), \
                   trimesh.Trimesh(vertices=z["target_vertices"],
                                   faces=z["target_faces"], process=False)
    return None, None


def render(mesh, centre, radius, size, elev=20.0, azim=35.0):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    yfov = np.pi / 5.0
    e, a = np.radians(elev), np.radians(azim)
    dist = 1.25 * radius / np.tan(yfov / 2)
    eye = centre + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    fwd = centre - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    pose = np.eye(4); pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, up, -fwd, eye
    sc = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.55, 0.55, 0.58])  # floats: ints render black
    for m in (mesh if isinstance(mesh, (list, tuple)) else [mesh]):
        sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    sc.add(pyrender.PerspectiveCamera(yfov=yfov), pose=pose)
    sc.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=pose)
    r = pyrender.OffscreenRenderer(*size)
    try:
        rgb, _ = r.render(sc)
    finally:
        r.delete()
    return Image.fromarray(rgb.astype(np.uint8))


def flat(mesh, col):
    m = mesh.copy()
    m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.tile(col, (len(m.faces), 1)))
    return m


def colour_leak(mesh, intr, thresh):
    """Red where the prediction sits on the intruder — the same test Leak% applies."""
    m = mesh.copy()
    ip, _ = trimesh.sample.sample_surface(intr, 40000, seed=5)
    d, _ = cKDTree(ip).query(m.vertices, k=1)
    vc = np.where((d <= thresh)[:, None], LEAK[None, :], GOOD[None, :]).astype(np.uint8)
    m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=vc)
    return m


def overlay(img_chw, mask_hw, colour=(214, 64, 40), alpha=0.45):
    a = np.clip(np.asarray(img_chw).transpose(1, 2, 0), 0, 1) * 255
    base = a.astype(np.uint8).copy()
    m = np.asarray(mask_hw) > 0.5
    tint = np.zeros_like(base); tint[:] = colour
    base[m] = (base[m] * (1 - alpha) + tint[m] * alpha).astype(np.uint8)
    return Image.fromarray(base)


def fig_masks(names, nviews=3, tile=190):
    rows = []
    for n in names:
        p = OUT / "masks" / f"{n}.npz"
        if p.exists():
            rows.append((n, np.load(p)))
    if not rows:
        return None
    labw, hdr, gap = 210, 92, 8
    W = labw + nviews * (tile * 3 + gap * 2 + 26)
    H = hdr + len(rows) * (tile + 30)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((16, 16), "What the model is told is the target", fill=INK, font=font(26, True))
    d.text((16, 50), "red = pixels marked foreground.  left: the photo   middle: mask built today"
                     "   right: mask after the fix", fill=MUTED, font=font(15))
    for r, (n, z) in enumerate(rows):
        y = hdr + r * (tile + 30)
        d.text((16, y + tile // 2 - 8), n.split("__")[1][:18], fill=INK, font=font(16, True))
        for vi in range(nviews):
            x = labw + vi * (tile * 3 + gap * 2 + 26)
            img = z["images"][vi]
            raw = Image.fromarray((np.clip(img.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8))
            sheet.paste(raw.resize((tile, tile)), (x, y))
            sheet.paste(overlay(img, z["stock"][vi, 0]).resize((tile, tile)), (x + tile + gap, y))
            sheet.paste(overlay(img, z["fixed"][vi, 0]).resize((tile, tile)),
                        (x + 2 * (tile + gap), y))
    return sheet


def fig_recon(names, cell=270):
    man = json.loads((OUT / "manifest.json").read_text())
    rows = [n for n in names if n in man]
    if not rows:
        return None
    labw, hdr, gap = 210, 96, 8
    cols = ["ground truth + intruder"] + [lbl for _a, lbl in ARMS]
    W = labw + len(cols) * (cell + gap)
    H = hdr + len(rows) * (cell + 34)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((16, 16), "What the model built", fill=INK, font=font(26, True))
    d.text((16, 50), "red = predicted surface sitting on the intruder — the same test Leak% counts",
           fill=MUTED, font=font(15))
    for i, c in enumerate(cols):
        x = labw + i * (cell + gap)
        tw = d.textlength(c, font=font(15, True))
        d.text((x + (cell - tw) / 2, hdr - 24), c, fill=MUTED, font=font(15, True))
    for r, n in enumerate(rows):
        y = hdr + r * (cell + 34)
        intr, tgt = intruder_mesh(n)
        if intr is None:
            continue
        allv = np.vstack([tgt.vertices, intr.vertices])
        centre = (allv.min(0) + allv.max(0)) / 2
        radius = float(np.linalg.norm(allv.max(0) - allv.min(0))) * 0.5
        thresh = 0.03 * float(np.max(tgt.extents))
        d.text((16, y + cell // 2 - 8), n.split("__")[1][:18], fill=INK, font=font(16, True))
        sheet.paste(render([flat(tgt, GT_C), flat(intr, INTR)], centre, radius, (cell, cell)),
                    (labw, y))
        for i, (arm, _lbl) in enumerate(ARMS):
            p = man[n].get(f"{arm}__s0")
            if p and Path(p).exists():
                m = trimesh.load(p, force="mesh")
                if len(getattr(m, "faces", [])):
                    sheet.paste(render(colour_leak(m, intr, thresh), centre, radius, (cell, cell)),
                                (labw + (i + 1) * (cell + gap), y))
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", default=None)
    args = ap.parse_args()
    man = json.loads((OUT / "manifest.json").read_text())
    names = args.objects or sorted(man)
    a = fig_masks(names)
    if a:
        a.save(OUT / "fig_masks.jpg", quality=94)
        print(f"wrote {OUT/'fig_masks.jpg'}")
    b = fig_recon(names)
    if b:
        b.save(OUT / "fig_recon.jpg", quality=94)
        print(f"wrote {OUT/'fig_recon.jpg'}")


if __name__ == "__main__":
    main()
