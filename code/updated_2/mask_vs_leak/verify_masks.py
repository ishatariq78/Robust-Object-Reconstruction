# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Verify the masks BEFORE trusting any conclusion drawn from them.

The oracle-mask experiment is only an upper bound if the oracle mask is actually what it claims:
the target's silhouette with the intruder excluded. And the SAM2 arm only tells us something if
we know whether SAM2 was including the pot or not — which was never checked.

Both are measured here against ground truth silhouettes rendered through the same cameras:

    target_hit    share of the TARGET's pixels the mask covers          (want high)
    pot_hit       share of the POT's pixels the mask covers             (want ~0 for oracle)
    pot_share     share of the MASK that is pot rather than target      (the contamination)

The pot silhouette is rendered from the distractor mesh stored in the clutter label, through the
batch's own cameras, at the same scale the dataset applies — so it lines up with the mask pixel
for pixel rather than approximately.

    python updated_2/mask_vs_leak/verify_masks.py --objects <clutter pkl> ...
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "updated_2"))
sys.path.insert(0, str(REPO / "updated_shapeR"))
sys.path.insert(0, str(REPO / "updated_shapeR/testing"))


def silhouette(mesh, batch, h, w):
    from augment import render_through_views
    ext = batch["camera_extrinsics"][0].float().cpu().numpy()
    itr = batch["camera_intrinsics"][0].float().cpu().numpy()
    _rgb, alpha, _d = render_through_views(mesh, ext, itr, h, w)
    return alpha > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", required=True)
    ap.add_argument("--sam2_model_id", default="facebook/sam2.1-hiera-base-plus")
    ap.add_argument("--save_fig", default=str(HERE / "outputs" / "fig_mask_verify.jpg"))
    args = ap.parse_args()

    import omegaconf
    from dataset.shaper_dataset import InferenceDataset
    from model.download import setup_checkpoints
    from mask_conditioning import compute_view_foreground_masks
    from detect_foreground_sam2 import SAM2ForegroundSegmenter
    from oracle_mask import oracle_masks

    setup_checkpoints()
    config = omegaconf.OmegaConf.load(str(REPO / "checkpoints/config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam2 = SAM2ForegroundSegmenter(args.sam2_model_id, device=str(device), dtype=torch.bfloat16)
    ds = InferenceDataset(config, paths=list(args.objects), override_num_views=8)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                                         collate_fn=ds.custom_collate)

    print(f"{'object':26}{'mask':>9}{'target_hit':>12}{'pot_hit':>10}{'pot_share':>12}")
    panels = []
    for bi, batch in enumerate(loader):
        name = batch["name"][0]
        pkl = Path(args.objects[bi])
        lab = pkl.with_name(pkl.stem + "__label.npz")
        if not lab.exists():
            print(f"  {name}: no label, skipped")
            continue
        z = np.load(lab)
        d = pickle.load(open(pkl, "rb"))
        scale = 0.9 / float(np.max(d["bounds"].numpy()))
        pot = trimesh.Trimesh(vertices=z["distractor_vertices"] * scale,
                              faces=z["distractor_faces"], process=False)
        gt = trimesh.Trimesh(vertices=np.asarray(batch["vertices"][0], np.float64),
                             faces=np.asarray(batch["faces"][0]), process=False)
        b, v, c, h, w = batch["images"].shape
        imgs = batch["images"][0].float().cpu().numpy()
        tgt_sil = silhouette(gt, batch, h, w)
        pot_sil = silhouette(pot, batch, h, w)

        batch_d = InferenceDataset.move_batch_to_device(batch, device, dtype=torch.bfloat16)
        sam = compute_view_foreground_masks(
            batch_d["images"].reshape(b * v, c, h, w).float(),
            batch_d["masks_ingest"].reshape(b * v, 1, h, w).float(),
            batch_d["boxes_ingest"].reshape(b * v, 2, 2), sam2, max_points=16
        ).float().cpu().numpy().reshape(v, h, w) > 0.5
        ora = oracle_masks(batch_d, gt, occluder=pot).float().cpu().numpy().reshape(v, h, w) > 0.5

        for lbl, m in (("SAM2", sam), ("oracle", ora)):
            th = float((m & tgt_sil).sum()) / max(tgt_sil.sum(), 1) * 100
            ph = float((m & pot_sil).sum()) / max(pot_sil.sum(), 1) * 100
            ps = float((m & pot_sil).sum()) / max(m.sum(), 1) * 100
            print(f"{name.split('__')[1][:24]:26}{lbl:>9}{th:>11.1f}%{ph:>9.1f}%{ps:>11.1f}%")
        panels.append((name, imgs, tgt_sil, pot_sil, sam, ora))

    # one visual per object so the numbers can be checked by eye
    try:
        from PIL import Image, ImageDraw
        tile, gap, labw, hdr = 190, 6, 150, 66
        W = labw + 5 * (tile + gap)
        H = hdr + len(panels) * (tile + 26)
        sheet = Image.new("RGB", (W, H), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        dr.text((14, 14), "Mask check: is the pot inside the mask?", fill=(20, 22, 28))
        dr.text((14, 34), "photo | target truth | pot truth | SAM2 mask | oracle mask",
                fill=(110, 116, 126))
        for r, (name, imgs, tgt, pot, sam, ora) in enumerate(panels):
            y = hdr + r * (tile + 26)
            dr.text((14, y + tile // 2), name.split("__")[1][:18], fill=(20, 22, 28))
            base = (np.clip(imgs[0].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
            if base.shape[2] == 1:
                base = np.repeat(base, 3, axis=2)
            cells = [base]
            for m, col in ((tgt[0], (60, 150, 100)), (pot[0], (210, 90, 60)),
                           (sam[0], (60, 110, 190)), (ora[0], (150, 100, 190))):
                c_img = base.copy()
                c_img[m] = (c_img[m] * 0.4 + np.array(col) * 0.6).astype(np.uint8)
                cells.append(c_img)
            for i, cimg in enumerate(cells):
                sheet.paste(Image.fromarray(cimg).resize((tile, tile)),
                            (labw + i * (tile + gap), y))
        sheet.save(args.save_fig, quality=94)
        print(f"\nwrote {args.save_fig}")
    except Exception as e:
        print(f"figure skipped: {e}")


if __name__ == "__main__":
    main()
