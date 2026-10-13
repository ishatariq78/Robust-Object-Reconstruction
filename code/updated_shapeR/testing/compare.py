# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Visual comparison sheets: ORIGINAL vs TRAINED, one row per object.

ONE camera per object, framed on that object's ground-truth mesh at metric scale, reused for
every render of it. A previous version framed the ground truth from a *scaled* mesh while the
reconstructions were metric, which made the GT look twice the size of every prediction — the
comparison was unreadable. Getting this wrong silently misleads, so it is fixed in one place.

    python updated_shapeR/testing/compare.py            # all conditions found
    python updated_shapeR/testing/compare.py --conditions clean onobject
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]          # HERE is a dir: [0]=updated_shapeR, [1]=repo root
OUT = HERE / "outputs"

TITLES = {"clean": "CLEAN photos — nothing changed",
          "occluded": "OCCLUDED photos — part of the object hidden",
          "onobject": "AN OBJECT STANDING ON IT"}
INK, GREEN, GREY, ACCENT, MUTED = (20, 22, 28), (40, 140, 90), (90, 95, 105), (200, 70, 40), (110, 116, 126)


def font(sz, bold=False):
    p = f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"
    try:
        return ImageFont.truetype(p, sz)
    except Exception:
        return None


def render(mesh, centre, radius, size=(430, 430), elev=20.0, azim=35.0):
    """Z-up orbit; distance solved from the FOV so the whole object fits."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender

    m = mesh.copy()
    m.visual = trimesh.visual.ColorVisuals(
        m, face_colors=np.tile(np.array([168, 175, 186, 255], np.uint8), (len(m.faces), 1)))
    yfov = np.pi / 5.0
    e, a = np.radians(elev), np.radians(azim)
    d = 1.25 * radius / np.tan(yfov / 2)
    eye = centre + d * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    fwd = centre - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    pose = np.eye(4); pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, up, -fwd, eye
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


def views_strip(name, cond, n=4, tile=200):
    """The photos the models were given, as one strip."""
    p = OUT / "views" / f"{name}__{cond}__images.npy"
    if not p.exists():
        return None
    imgs = np.load(p)[:n]
    strip = Image.new("RGB", (tile * len(imgs), tile), (18, 18, 18))
    for i, im in enumerate(imgs):
        a = (np.clip(im.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        strip.paste(Image.fromarray(a).resize((tile, tile)), (i * tile, 0))
    return strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--cell", type=int, default=330)
    args = ap.parse_args()

    manifest = json.loads((OUT / "manifest.json").read_text())
    names = sorted(manifest)
    conds = args.conditions or [c for c in ("clean", "occluded", "onobject")
                                if any(f"{c}__original" in manifest[n].get("meshes", {}) for n in names)]
    sheets = OUT / "comparison"; sheets.mkdir(parents=True, exist_ok=True)

    # one camera per object, from its GT mesh at metric scale
    cams, gts = {}, {}
    for n in names:
        d = pickle.load(open(REPO / f"data/{n}.pkl", "rb"))
        gt = trimesh.Trimesh(vertices=d["mesh_vertices"].numpy().astype(np.float64),
                             faces=d["mesh_faces"].numpy(), process=False)
        gts[n] = gt
        cams[n] = (gt.bounds.mean(axis=0), float(np.linalg.norm(gt.extents)) * 0.5)

    for cond in conds:
        cell, lab, strip_w = args.cell, 210, args.cell * 2
        hdr = 62
        rows = [n for n in names if f"{cond}__original" in manifest[n].get("meshes", {})]
        if not rows:
            continue
        W = lab + strip_w + cell * 3
        H = hdr + 36 + cell * len(rows)
        sheet = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        d.text((16, 16), TITLES.get(cond, cond), fill=INK, font=font(28, True))
        for x, cap, col in ((lab, "the photos given to both", MUTED),
                            (lab + strip_w, "ground truth", GREEN),
                            (lab + strip_w + cell, "ORIGINAL (no mask)", GREY),
                            (lab + strip_w + cell * 2, "TRAINED (mask)", ACCENT)):
            wide = strip_w if "photos" in cap else cell
            tw = d.textlength(cap, font=font(17, True))
            d.text((x + (wide - tw) / 2, hdr + 6), cap, fill=col, font=font(17, True))

        for r, n in enumerate(rows):
            y = hdr + 36 + r * cell
            d.text((16, y + cell // 2 - 22), n.split("__")[-1], fill=INK, font=font(19, True))
            s = manifest[n].get("stats", {}).get(cond, {})
            note = ("" if cond == "clean" else
                    (f"{s.get('achieved', 0)*100:.0f}% hidden" if cond == "occluded"
                     else f"object = {s.get('achieved_area', 0)*100:.0f}% of it"))
            if note:
                d.text((16, y + cell // 2 + 4), note, fill=MUTED, font=font(14))
            st = views_strip(n, cond, n=4, tile=cell // 2)
            if st:
                sheet.paste(st, (lab, y + (cell - st.height) // 2))
            centre, radius = cams[n]
            sheet.paste(render(gts[n], centre, radius, (cell, cell)), (lab + strip_w, y))
            for i, model in enumerate(("original", "trained")):
                mp = manifest[n]["meshes"].get(f"{cond}__{model}")
                if mp and Path(mp).exists():
                    sheet.paste(render(trimesh.load(mp, force="mesh"), centre, radius, (cell, cell)),
                                (lab + strip_w + cell * (i + 1), y))
        p = sheets / f"{cond}.jpg"
        sheet.save(p, quality=94)
        print(f"  wrote {p}")

    print(f"\nSheets in {sheets}")


if __name__ == "__main__":
    main()
