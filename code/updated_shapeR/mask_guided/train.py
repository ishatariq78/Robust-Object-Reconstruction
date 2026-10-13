# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Fine-tune the mask-guidance parameters of ShapeR (Option A mask channel + Option B
attention-bias scale), with the whole backbone FROZEN. Zero-init means training starts
from the exact pretrained behaviour and only *adds* mask guidance.

Run from the ShapeR repo root (needs model/, dataset/, checkpoints/):

    python updated_shapeR/mask_guided/train.py \
        --data_pkls data/ADT1292__stool.pkl data/BNB1045__sofa.pkl ... \
        --num_views 8 --epochs 20 --lr 1e-3 \
        --out_ckpt updated_shapeR/outputs/mask_guided.ckpt

Only a few thousand parameters are trained (lambda + the widened mask conv), so this is
closer to LoRA than a full retrain. See train_mask_guided.slurm for the cluster command.

VERIFY ON FIRST RUN (honest): the target-latent step (VAE `encode_for_diffusion`) has
assumptions about surface-point count and token scale — the script asserts z1's shape and
prints it so a mismatch is a clear error, not a silent bug.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import omegaconf
import torch
from torchsparse import SparseTensor
from tqdm import tqdm

# repo root + updated_shapeR on path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "updated_shapeR"))

from dataset.shaper_dataset import InferenceDataset          # noqa: E402
from model.download import setup_checkpoints                 # noqa: E402
from model.text.hf_embedder import TextFeatureExtractor      # noqa: E402
from model.vae3d.autoencoder import MichelangeloLikeAutoencoderWrapper  # noqa: E402

from mask_conditioning import compute_view_foreground_masks  # noqa: E402
from mask_guided.architecture import build_mask_guided_denoiser, trainable_parameters  # noqa: E402
from mask_guided.losses import flow_matching_loss, encode_target_latent, background_occupancy_loss  # noqa: E402


def build_sam2(model_id, device):
    from detect_foreground_sam2 import SAM2ForegroundSegmenter
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    return SAM2ForegroundSegmenter(model_id, device=str(device), dtype=dtype)


def compute_fg_for_batch(batch, sam2, max_points=16):
    b, v, c, h, w = batch["images"].shape
    imgs = batch["images"].reshape(b * v, c, h, w).float()
    pmask = batch["masks_ingest"].reshape(b * v, 1, h, w).float()
    boxes = batch["boxes_ingest"].reshape(b * v, 2, 2)
    return compute_view_foreground_masks(imgs, pmask, boxes, sam2, max_points=max_points)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_pkls", nargs="+", required=True, help="Training pkl samples (must contain GT meshes).")
    ap.add_argument("--num_views", type=int, default=8)          # Exp 3: 8 is the sweet spot
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--token_prob_index", type=int, default=6)   # matches inference scale_prob[6]
    ap.add_argument("--surface_points", type=int, default=16384)
    ap.add_argument("--mask_loss_weight", type=float, default=0.0, help=">0 enables background-occupancy loss (experimental).")
    ap.add_argument("--no_fg_channel", action="store_true", help="Disable Option A (train only the attention bias).")
    ap.add_argument("--no_fg_bias", action="store_true", help="Disable Option B (train only the mask channel).")
    ap.add_argument("--sam2_model_id", default="facebook/sam2.1-hiera-base-plus")
    ap.add_argument("--out_ckpt", default="updated_shapeR/outputs/mask_guided.ckpt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_checkpoints()
    config = omegaconf.OmegaConf.load("checkpoints/config.yaml")

    print("Building mask-guided model (pretrained + Option A/B, zero-init)...")
    model, missing, unexpected = build_mask_guided_denoiser(
        config, "checkpoints/019-0-bfloat16.ckpt", device,
        use_fg_channel=not args.no_fg_channel, use_fg_bias=not args.no_fg_bias,
    )
    print(f"  loaded pretrained (missing={len(missing)} new keys, unexpected={len(unexpected)})")

    vae = MichelangeloLikeAutoencoderWrapper("checkpoints/vae-088-0-bfloat16.ckpt", device)
    vae.model = vae.model.float()   # encoder's fps()/torch_cluster op is not implemented for bfloat16
    text_fx = TextFeatureExtractor(device=device).to(torch.bfloat16)
    sam2 = build_sam2(args.sam2_model_id, device)

    params = trainable_parameters(model, train_bias=not args.no_fg_bias, train_mask_channel=not args.no_fg_channel)
    n_train = sum(p.numel() for p in params)
    print(f"  trainable params: {n_train} across {len(params)} tensors (backbone frozen)")
    opt = torch.optim.AdamW(params, lr=args.lr)

    ds = InferenceDataset(config, paths=list(args.data_pkls), override_num_views=args.num_views)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=0,
                                         collate_fn=ds.custom_collate)

    model.train()
    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        running = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            if "vertices" not in batch:
                continue  # need GT mesh to build a target
            batch = InferenceDataset.move_batch_to_device(batch, device, dtype=torch.bfloat16)
            batch["t5_text"], batch["clip_text"] = text_fx(batch["caption"])
            pc = batch["semi_dense_points"]
            batch["semi_dense_points"] = SparseTensor(pc.feats, pc.coords, pc.stride)

            fg = compute_fg_for_batch(batch, sam2, max_points=16).to(device=device, dtype=torch.bfloat16)
            model._fg_masks = fg

            z1 = encode_target_latent(
                vae, batch["vertices"][0], batch["faces"][0], scale=1.0,   # vertices already scaled by dataset
                token_prob_index=args.token_prob_index, n_points=args.surface_points,
                device=device, dtype=torch.float32,   # encoder runs in fp32; cast target to bf16 below
            )
            assert z1.dim() == 3 and z1.shape[-1] == vae.get_embed_dim(), (
                f"target latent shape {tuple(z1.shape)} unexpected; check token_prob_index / VAE encode")
            z1 = z1.to(torch.bfloat16)   # flow model is bfloat16

            loss = flow_matching_loss(model, batch, z1)
            if args.mask_loss_weight > 0:
                # recompute one z_t/v_pred pair for the silhouette term (experimental)
                B, L, D = z1.shape
                z0 = torch.randn_like(z1)
                t = torch.rand(1, device=device)
                z_t = (1 - t.view(1, 1, 1).to(z1.dtype)) * z0 + t.view(1, 1, 1).to(z1.dtype) * z1
                v_pred = model(z_t, t.to(z1.dtype), batch)
                loss = loss + args.mask_loss_weight * background_occupancy_loss(
                    model, vae, batch, z_t, v_pred, t, fg)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            model._fg_masks = None
            running += float(loss.detach())

        print(f"epoch {epoch+1}: mean loss={running/max(1,len(loader)):.4f}  "
              f"fg_bias_scale={float(model.fg_bias_scale):.4f}")
        torch.save({
            "fg_bias_scale": model.fg_bias_scale.detach().cpu(),
            "mask_conv": model.dino_ray_extractor._mask_patch_embed[0].proj_in.state_dict()
                         if not args.no_fg_channel else None,
            "args": vars(args),
        }, args.out_ckpt)
    print(f"Saved mask-guidance weights to {args.out_ckpt}")


if __name__ == "__main__":
    main()
