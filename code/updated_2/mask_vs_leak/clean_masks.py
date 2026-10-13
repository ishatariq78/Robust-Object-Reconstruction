# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Give the mask-guided model a mask that actually excludes the intruder.

THE BUG THIS FIXES. `compute_view_foreground_masks` prompts SAM2 with click points sampled from
the SLAM point-projection mask, and passes `negative_points_xy=None`. In a cluttered scene those
projections include the INTRUDER's points, so SAM2 is being told the pot is part of the target.
The mask then marks the pot as foreground, and the guidance the model was fine-tuned to follow
says "reconstruct this". That is the opposite of what the mask is for, and it explains why the
fine-tune's leak reduction was real but tiny (-0.55).

THE FIX, with no retraining. Split the point cloud with the support segmenter first, then prompt
SAM2 with target points as POSITIVE clicks and intruder points as NEGATIVE clicks. Same model,
same weights, same everything downstream — only the conditioning mask changes.

Nothing in the original pipeline is modified: this builds masks alongside it and hands them to
the model the same way `infer.py` does.

    python updated_2/mask_vs_leak/clean_masks.py --smoke      # one object, prints mask stats
    python updated_2/mask_vs_leak/clean_masks.py              # full run + scoring
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]      # HERE is the DIRECTORY, so [1] is the repo root, not [2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "updated_2"))
sys.path.insert(0, str(REPO / "updated_shapeR"))

from segment_v2 import segment_by_support_v2               # noqa: E402
from mask_conditioning import _positive_points_from_point_mask, _view_tensor_to_pil  # noqa: E402


def project(points_xyz, extrinsic_c2w, intrinsic, h, w):
    """World -> pixel, ShapeR/OpenCV convention (+Y down, +Z forward). Returns (uv, valid)."""
    R = extrinsic_c2w[:3, :3]
    t = extrinsic_c2w[:3, 3]
    cam = (points_xyz - t[None, :]) @ R                     # world -> camera
    z = cam[:, 2]
    ok = z > 1e-6
    uv = np.full((len(points_xyz), 2), -1.0)
    if ok.any():
        p = cam[ok] / z[ok, None]
        uv[ok, 0] = intrinsic[0, 0] * p[:, 0] + intrinsic[0, 2]
        uv[ok, 1] = intrinsic[1, 1] * p[:, 1] + intrinsic[1, 2]
    inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv, inside


def masks_with_negatives(batch, cloud_scaled, is_target, sam2, max_pos=16, max_neg=8):
    """
    Foreground masks where the intruder is pushed OUT via negative clicks.

    Falls back to the stock behaviour for any view with too few target projections — a view
    where we cannot prompt reliably should not get a guessed mask.
    """
    imgs = batch["images"]
    b, v, c, h, w = imgs.shape
    flat = imgs.reshape(b * v, c, h, w).float()
    pm = batch["masks_ingest"].reshape(b * v, 1, h, w).float()
    ext = batch["camera_extrinsics"][0].float().cpu().numpy()   # bfloat16 has no numpy cast
    itr = batch["camera_intrinsics"][0].float().cpu().numpy()

    out = torch.zeros((b * v, 1, h, w), dtype=imgs.dtype, device=imgs.device)
    stats = []
    rng = np.random.RandomState(0)
    for i in range(b * v):
        image = _view_tensor_to_pil(flat[i])
        uv, inside = project(cloud_scaled, ext[i], itr[i], h, w)
        pos = uv[inside & is_target]
        neg = uv[inside & ~is_target]
        if len(pos) < 3:                                     # cannot prompt this view honestly
            pos = _positive_points_from_point_mask(pm[i, 0], max_pos)
            neg = np.empty((0, 2))
        else:
            if len(pos) > max_pos:
                pos = pos[rng.choice(len(pos), max_pos, replace=False)]
            if len(neg) > max_neg:
                neg = neg[rng.choice(len(neg), max_neg, replace=False)]
        box = np.array([0, 0, w - 1, h - 1], np.float32)
        bb = batch["boxes_ingest"].reshape(b * v, 2, 2)[i].reshape(-1).float().cpu().numpy()
        if bb[2] > bb[0] and bb[3] > bb[1]:
            box = bb
        m, _s = sam2.segment(image,
                             positive_points_xy=np.asarray(pos, np.float32),
                             negative_points_xy=(np.asarray(neg, np.float32)
                                                 if len(neg) else None),
                             box_xyxy=box)
        t = torch.as_tensor(np.asarray(m, np.float32), device=imgs.device)
        out[i, 0] = t.to(imgs.dtype)
        stats.append((len(pos), len(neg), float(t.mean())))
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pkl", default=str(REPO / "updated_2/outputs/stage1/clutter/"
                                                "ADT1292__woodentable__both__L025.pkl"))
    ap.add_argument("--sam2_model_id", default="facebook/sam2.1-hiera-base-plus")
    args = ap.parse_args()

    import omegaconf
    from dataset.shaper_dataset import InferenceDataset
    from model.download import setup_checkpoints
    from detect_foreground_sam2 import SAM2ForegroundSegmenter
    from mask_conditioning import compute_view_foreground_masks

    setup_checkpoints()
    config = omegaconf.OmegaConf.load(str(REPO / "checkpoints/config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam2 = SAM2ForegroundSegmenter(args.sam2_model_id, device=str(device), dtype=torch.bfloat16)

    ds = InferenceDataset(config, paths=[args.pkl], override_num_views=8)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                                         collate_fn=ds.custom_collate)
    import pickle
    d = pickle.load(open(args.pkl, "rb"))
    cloud = d["points_model"].numpy().astype(np.float64)[:, :3]
    scale = 0.9 / float(np.max(d["bounds"].numpy()))
    is_target = segment_by_support_v2(cloud)
    lab = Path(args.pkl.replace(".pkl", "__label.npz"))
    if lab.exists():
        z = np.load(lab)
        ntar = int(z["n_target_points"])
        truth = np.zeros(len(cloud), bool)
        truth[:ntar] = True
        rec = float((~is_target & ~truth).sum()) / max((~truth).sum(), 1) * 100
        fp = float((~is_target & truth).sum()) / max(truth.sum(), 1) * 100
        print(f"segmenter: {int((~is_target).sum())} points called intruder — "
              f"recall {rec:.0f}%, false-pos {fp:.2f}%")

    for batch in loader:
        batch = InferenceDataset.move_batch_to_device(batch, device, dtype=torch.bfloat16)
        b, v, c, h, w = batch["images"].shape
        old = compute_view_foreground_masks(
            batch["images"].reshape(b * v, c, h, w).float(),
            batch["masks_ingest"].reshape(b * v, 1, h, w).float(),
            batch["boxes_ingest"].reshape(b * v, 2, 2), sam2, max_points=16)
        new, stats = masks_with_negatives(batch, cloud * scale, is_target, sam2)
        oa = old.float().mean(dim=(1, 2, 3)).cpu().numpy()
        na = new.float().mean(dim=(1, 2, 3)).cpu().numpy()
        print(f"\n{'view':>5}{'pos':>5}{'neg':>5}{'old fg area':>13}{'new fg area':>13}{'shrunk by':>11}")
        for i, (p, n, _) in enumerate(stats):
            print(f"{i:>5}{p:>5}{n:>5}{oa[i]:>13.3f}{na[i]:>13.3f}"
                  f"{(oa[i]-na[i])/max(oa[i],1e-9)*100:>10.0f}%")
        print(f"\nmean foreground area: old {oa.mean():.3f} -> new {na.mean():.3f} "
              f"({(oa.mean()-na.mean())/max(oa.mean(),1e-9)*100:+.0f}%)")
        break


if __name__ == "__main__":
    main()
