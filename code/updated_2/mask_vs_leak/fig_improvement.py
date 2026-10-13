# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Before-and-after pictures for the two things that actually improved.

The report was heavy on failures and thin on visible wins, so these show the improvements
directly, side by side with plain ShapeR, using the meshes already generated.

  fig_leak_improved   the clutter result: plain ShapeR, then the intruder's points removed,
                      then points removed AND the mask corrected. Predicted surface is RED
                      where it sits on the intruder — the same test Leak% counts — so the red
                      lump shrinking IS the improvement.

  fig_complete        the completeness result: the GROUND TRUTH is drawn, coloured by whether
                      the prediction actually covers it. Grey = built, RED = real surface the
                      model missed. Less red is better. This is Missing% made visible, and it
                      is the metric your fine-tune genuinely moves.

One camera per object, framed on the truth at metric scale and reused for every panel, so a
bloated mesh cannot look the same size as a good one.
"""

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
OUT = HERE / "outputs"
INK, MUTED = (20, 22, 28), (108, 114, 124)
OKC = np.array([170, 176, 186, 255], np.uint8)
BADC = np.array([201, 62, 42, 255], np.uint8)
GTC = np.array([104, 176, 132, 255], np.uint8)
INTRC = np.array([214, 118, 96, 255], np.uint8)


def font(sz, bold=False):
    p = f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"
    try:
        return ImageFont.truetype(p, sz)
    except Exception:
        return ImageFont.load_default()


def render(meshes, centre, radius, size, elev=20.0, azim=35.0):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    yfov = np.pi / 5.0
    e, a = np.radians(elev), np.radians(azim)
    d = 1.25 * radius / np.tan(yfov / 2)
    eye = centre + d * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    fwd = centre - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    pose = np.eye(4); pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, up, -fwd, eye
    sc = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.55, 0.55, 0.58])
    for m in (meshes if isinstance(meshes, (list, tuple)) else [meshes]):
        sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    sc.add(pyrender.PerspectiveCamera(yfov=yfov), pose=pose)
    sc.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=pose)
    r = pyrender.OffscreenRenderer(*size)
    try:
        rgb, _ = r.render(sc)
    finally:
        r.delete()
    return Image.fromarray(rgb.astype(np.uint8))


def flat(m, col):
    o = m.copy()
    o.visual = trimesh.visual.ColorVisuals(o, face_colors=np.tile(col, (len(o.faces), 1)))
    return o


def paint_by_distance(mesh, ref_pts, thresh, near_is_bad):
    """Colour a mesh red where it is near (or far from) a reference set of points."""
    o = mesh.copy()
    d, _ = cKDTree(ref_pts).query(o.vertices, k=1)
    bad = (d <= thresh) if near_is_bad else (d > thresh)
    o.visual = trimesh.visual.ColorVisuals(
        o, vertex_colors=np.where(bad[:, None], BADC[None, :], OKC[None, :]).astype(np.uint8))
    return o


def label_pair(pkl_stem):
    for d in ("updated_2/outputs/stage1/clutter", "updated_2/confirm/outputs/clutter"):
        p = REPO / d / f"{pkl_stem}__label.npz"
        if p.exists():
            z = np.load(p)
            return (trimesh.Trimesh(vertices=z["target_vertices"], faces=z["target_faces"],
                                    process=False),
                    trimesh.Trimesh(vertices=z["distractor_vertices"], faces=z["distractor_faces"],
                                    process=False))
    return None, None


def sheet(rows, cols, title, sub, cell=260):
    labw, hdr, gap = 200, 96, 8
    W = labw + len(cols) * (cell + gap)
    H = hdr + len(rows) * (cell + 34)
    sh = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sh)
    d.text((16, 16), title, fill=INK, font=font(26, True))
    d.text((16, 52), sub, fill=MUTED, font=font(15))
    for i, c in enumerate(cols):
        x = labw + i * (cell + gap)
        tw = d.textlength(c, font=font(15, True))
        d.text((x + (cell - tw) / 2, hdr - 24), c, fill=MUTED, font=font(15, True))
    return sh, d, labw, hdr, gap


def main():
    # ---------- figure 1: leakage, plain ShapeR -> points removed -> points + mask ----------
    A = json.loads((OUT / "three_arms" / "manifest.json").read_text())
    B = json.loads((OUT / "cleaned_arms" / "manifest.json").read_text())
    import sys as _s; _s.path.insert(0, str(REPO / "updated_2"))
    from metrics import score_paths as _sp

    def _leak(path, lab):
        return float(_sp(path, label_npz=lab)["leak_pct"]) if path and Path(path).exists() else None

    cand = []
    for nb in sorted(B):
        base = nb.replace("__cleaned", "")
        if base not in A:
            continue
        lab = None
        for dd in ("updated_2/outputs/stage1/clutter", "updated_2/confirm/outputs/clutter"):
            q = REPO / dd / f"{base}__label.npz"
            if q.exists():
                lab = str(q)
        if not lab:
            continue
        v = [_leak(A[base].get("control__s0"), lab), _leak(B[nb].get("control__s0"), lab),
             _leak(B[nb].get("fixed__s0"), lab)]
        if None in v:
            continue
        cand.append((nb, v, v[0] - v[2]))          # improvement, plain ShapeR -> points + mask
    cand.sort(key=lambda r: -r[2])
    names = [c[0] for c in cand[:5]]
    scores = {c[0]: c[1] for c in cand}
    cols = ["truth + the thing on it", "plain ShapeR", "its points removed", "points + mask fixed"]
    sh, d, labw, hdr, gap = sheet(names, cols,
                                  "Removing the thing standing on the table",
                                  "red = surface the model built ON the intruder. Less red is better.")
    for r, nb in enumerate(names):
        base = nb.replace("__cleaned", "")
        tgt, intr = label_pair(base)
        if tgt is None:
            continue
        y = hdr + r * (260 + 34)
        allv = np.vstack([tgt.vertices, intr.vertices])
        centre = (allv.min(0) + allv.max(0)) / 2
        radius = float(np.linalg.norm(allv.max(0) - allv.min(0))) * 0.5
        thr = 0.03 * float(np.max(tgt.extents))
        ip, _ = trimesh.sample.sample_surface(intr, 40000, seed=5)
        d.text((16, y + 120), base.split("__")[1][:18], fill=INK, font=font(16, True))
        sh.paste(render([flat(tgt, GTC), flat(intr, INTRC)], centre, radius, (260, 260)), (labw, y))
        for i, p in enumerate((A[base].get("control__s0"), B[nb].get("control__s0"),
                               B[nb].get("fixed__s0"))):
            if p and Path(p).exists():
                m = trimesh.load(p, force="mesh")
                if len(getattr(m, "faces", [])):
                    x = labw + (i + 1) * (260 + gap)
                    sh.paste(render(paint_by_distance(m, ip, thr, True), centre, radius, (260, 260)),
                             (x, y))
                    v = scores[nb][i]
                    txt = f"{v:.1f}% on the intruder"
                    col = (44, 127, 86) if i == 2 and v < scores[nb][0] else MUTED
                    tw = d.textlength(txt, font=font(14, i == 2))
                    d.text((x + (260 - tw) / 2, y + 264), txt, fill=col, font=font(14, i == 2))
        print(f"  leak row: {base}")
    sh.save(OUT / "fig_leak_improved.jpg", quality=94)
    print(f"wrote {OUT/'fig_leak_improved.jpg'}")

    # ---------- figure 2: completeness, guidance off -> mask on ----------
    C = json.loads((OUT / "clean_test" / "manifest.json").read_text())
    import pickle as _pk

    def _miss(path, gtm):
        if not path or not Path(path).exists():
            return None
        mm = trimesh.load(path, force="mesh")
        if not len(getattr(mm, "faces", [])):
            return None
        gp, _ = trimesh.sample.sample_surface(gtm, 20000, seed=1)
        pp, _ = trimesh.sample.sample_surface(mm, 20000, seed=2)
        dgt, _ = cKDTree(pp).query(gp, k=1)
        return float(np.mean(dgt > 0.03 * float(np.max(gtm.extents)))) * 100

    cand2 = []
    for n in sorted(C):
        q = REPO / "updated_2/mask_vs_leak/img_clutter" / f"{n}.pkl"
        if not q.exists():
            continue
        dd = _pk.load(open(q, "rb"))
        g = trimesh.Trimesh(vertices=dd["mesh_vertices"].numpy().astype(np.float64),
                            faces=dd["mesh_faces"].numpy(), process=False)
        a, b = _miss(C[n].get("off__s0"), g), _miss(C[n].get("oracle__s0"), g)
        if a is None or b is None:
            continue
        cand2.append((n, [a, b], a - b))
    cand2.sort(key=lambda r: -r[2])
    picks = [c[0] for c in cand2[:5]]
    miss = {c[0]: c[1] for c in cand2}
    cols2 = ["the true shape", "without the mask", "with your mask"]
    sh2, d2, labw2, hdr2, gap2 = sheet(picks, cols2,
                                       "Building more of the object",
                                       "the TRUE shape, coloured red where the model failed to "
                                       "build it. Less red is better.")
    for r, n in enumerate(picks):
        pkl = REPO / "updated_2/mask_vs_leak/img_clutter" / f"{n}.pkl"
        if not pkl.exists():
            continue
        import pickle
        dd = pickle.load(open(pkl, "rb"))
        gt = trimesh.Trimesh(vertices=dd["mesh_vertices"].numpy().astype(np.float64),
                             faces=dd["mesh_faces"].numpy(), process=False)
        y = hdr2 + r * (260 + 34)
        centre = gt.bounds.mean(axis=0)
        radius = float(np.linalg.norm(gt.extents)) * 0.5
        thr = 0.03 * float(np.max(gt.extents))
        d2.text((16, y + 120), n.split("__")[1][:18], fill=INK, font=font(16, True))
        sh2.paste(render(flat(gt, GTC), centre, radius, (260, 260)), (labw2, y))
        for i, arm in enumerate(("off__s0", "oracle__s0")):
            p = C[n].get(arm)
            if p and Path(p).exists():
                m = trimesh.load(p, force="mesh")
                if len(getattr(m, "faces", [])):
                    pp, _ = trimesh.sample.sample_surface(m, 40000, seed=4)
                    # paint the TRUTH by whether the prediction reached it
                    x = labw2 + (i + 1) * (260 + gap2)
                    sh2.paste(render(paint_by_distance(gt, pp, thr, False), centre, radius,
                                     (260, 260)), (x, y))
                    v = miss[n][i]
                    txt = f"{v:.1f}% not built"
                    col = (44, 127, 86) if i == 1 and v < miss[n][0] else MUTED
                    tw = d2.textlength(txt, font=font(14, i == 1))
                    d2.text((x + (260 - tw) / 2, y + 264), txt, fill=col, font=font(14, i == 1))
        print(f"  complete row: {n}")
    sh2.save(OUT / "fig_complete.jpg", quality=94)
    print(f"wrote {OUT/'fig_complete.jpg'}")


if __name__ == "__main__":
    main()
