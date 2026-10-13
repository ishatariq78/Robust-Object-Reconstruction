# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Score held-out reconstructions against the GT mesh (taken from each pkl).

  Extra%   = fraction of predicted surface far (> 3% of extent) from the GT surface — the
             "extra / background / hallucinated geometry" we want to reduce (lower = better).
  Chamfer% = symmetric mean surface distance as % of object extent (lower = better).

The saved .glb from infer.py is in the object/model frame at metric scale (rescale_back with
do_transform_to_world=False), which matches the pkl's raw mesh_vertices/faces — so we compare
directly.

    python updated_shapeR/mask_guided/score_heldout.py \
        --input_pkls data/ADT1292__stool.pkl ... \
        --baseline_dir updated_shapeR/outputs/recon_heldout_baseline \
        --guided_dir   updated_shapeR/outputs/recon_heldout_masked
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def gt_mesh_from_pkl(pkl_path):
    d = pickle.load(open(pkl_path, "rb"))
    v = d["mesh_vertices"].numpy().astype(np.float64)
    f = d["mesh_faces"].numpy()
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def scores(pred_path, gt_mesh, thresh=0.03, n=8000):
    if not Path(pred_path).exists():
        return None
    pred = trimesh.load(pred_path, force="mesh")
    if len(pred.faces) == 0 or len(gt_mesh.faces) == 0:
        return None
    pp, _ = trimesh.sample.sample_surface(pred, n)
    gp, _ = trimesh.sample.sample_surface(gt_mesh, n)
    ext = float(np.max(gt_mesh.extents))
    d_pred, _ = cKDTree(gp).query(pp, k=1)   # pred -> gt (extra geometry)
    d_gt, _ = cKDTree(pp).query(gp, k=1)     # gt -> pred (missing geometry)
    extra = float(np.mean(d_pred > thresh * ext)) * 100
    chamfer = (d_pred.mean() + d_gt.mean()) / 2 / ext * 100
    return extra, chamfer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_pkls", nargs="+", required=True)
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--guided_dir", required=True)
    args = ap.parse_args()

    print(f"\n{'sample':30s} {'Extra% base':>11} {'Extra% guided':>14} {'Δ':>7}   "
          f"{'Cham base':>9} {'Cham guided':>11}")
    print("-" * 92)
    e_b, e_g, n_ok, n_better = [], [], 0, 0
    for pkl in args.input_pkls:
        name = Path(pkl).stem
        gt = gt_mesh_from_pkl(pkl)
        sb = scores(Path(args.baseline_dir) / f"{name}.glb", gt)
        sg = scores(Path(args.guided_dir) / f"{name}.glb", gt)
        if sb is None or sg is None:
            print(f"{name:30s}   (missing mesh: base={sb is not None}, guided={sg is not None})")
            continue
        d = sg[0] - sb[0]
        n_ok += 1; n_better += int(d < 0)
        e_b.append(sb[0]); e_g.append(sg[0])
        print(f"{name:30s} {sb[0]:>11.1f} {sg[0]:>14.1f} {d:>+7.1f}   {sb[1]:>9.2f} {sg[1]:>11.2f}")

    if n_ok:
        print("-" * 92)
        print(f"{'MEAN':30s} {np.mean(e_b):>11.1f} {np.mean(e_g):>14.1f} "
              f"{np.mean(e_g)-np.mean(e_b):>+7.1f}")
        print(f"\nMask-guided lowered Extra% on {n_better}/{n_ok} held-out samples "
              f"(mean {np.mean(e_b):.1f}% → {np.mean(e_g):.1f}%).")
        print("Note: single stochastic run each; run a few seeds to firm up the size of the effect.")


if __name__ == "__main__":
    main()
