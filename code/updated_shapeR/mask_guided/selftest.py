# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Correctness self-test for the mask-guided model. Run this FIRST on a GPU.

It proves two things on one real sample:
  (1) ZERO-INIT IDENTITY — with fg_bias_scale=0 and the zero-init mask channel, the
      mask-guided model reproduces base ShapeR's velocity (up to attention-kernel
      numerics). i.e. we did not break the pretrained model.
  (2) MECHANISM IS LIVE — setting fg_bias_scale to a non-zero value CHANGES the output,
      proving the foreground attention bias (Option B) is actually wired into attention.

    python updated_shapeR/mask_guided/selftest.py --pkl data/BNB1045__sofa.pkl
"""

import argparse
import sys
from pathlib import Path

import omegaconf
import torch
from torchsparse import SparseTensor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "updated_shapeR"))

from dataset.shaper_dataset import InferenceDataset
from model.download import setup_checkpoints
from model.flow_matching.shaper_denoiser import ShapeRDenoiser
from model.text.hf_embedder import TextFeatureExtractor

from mask_conditioning import compute_view_foreground_masks
from mask_guided.architecture import build_mask_guided_denoiser


def make_batch(pkl, config, num_views, device, text_fx, sam2):
    ds = InferenceDataset(config, paths=[pkl], override_num_views=num_views)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=ds.custom_collate)
    batch = next(iter(loader))
    batch = InferenceDataset.move_batch_to_device(batch, device, dtype=torch.bfloat16)
    batch["t5_text"], batch["clip_text"] = text_fx(batch["caption"])
    b, v, c, h, w = batch["images"].shape
    fg = compute_view_foreground_masks(
        batch["images"].reshape(b * v, c, h, w).float(),
        batch["masks_ingest"].reshape(b * v, 1, h, w).float(),
        batch["boxes_ingest"].reshape(b * v, 2, 2), sam2, max_points=16,
    ).to(device=device, dtype=torch.bfloat16)
    return batch, fg


def fresh_pc(batch):
    pc = batch["semi_dense_points"]
    batch["semi_dense_points"] = SparseTensor(pc.feats, pc.coords, pc.stride)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/BNB1045__sofa.pkl")
    ap.add_argument("--num_views", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_checkpoints()
    config = omegaconf.OmegaConf.load("checkpoints/config.yaml")
    text_fx = TextFeatureExtractor(device=device).to(torch.bfloat16)
    from detect_foreground_sam2 import SAM2ForegroundSegmenter
    sam2 = SAM2ForegroundSegmenter("facebook/sam2.1-hiera-base-plus", device=str(device), dtype=torch.bfloat16)

    # base ShapeR
    base = ShapeRDenoiser(config).to(device); base.convert_to_bfloat16()
    base.load_state_dict(torch.load("checkpoints/019-0-bfloat16.ckpt", map_location=device, weights_only=False), strict=False)
    base.eval()

    # mask-guided (zero-init)
    guided, _, _ = build_mask_guided_denoiser(config, "checkpoints/019-0-bfloat16.ckpt", device)
    guided.eval()

    batch, fg = make_batch(args.pkl, config, args.num_views, device, text_fx, sam2)
    L = 512  # arbitrary latent-token count for the identity check
    x = torch.randn(1, L, config.vae.embed_dim, device=device, dtype=torch.bfloat16)
    t = torch.tensor([0.5], device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        fresh_pc(batch); out_base = base(x, t, batch).float()
        guided._fg_masks = fg
        guided.fg_bias_scale.data.zero_()
        fresh_pc(batch); out_g0 = guided(x, t, batch).float()
        guided.fg_bias_scale.data.fill_(2.0)
        fresh_pc(batch); out_g1 = guided(x, t, batch).float()

    d_identity = (out_g0 - out_base).abs()
    d_effect = (out_g1 - out_g0).abs()
    scale = out_base.abs().mean().item()
    print("\n==== mask-guided self-test ====")
    print(f"output scale (mean|base|)          : {scale:.4f}")
    print(f"(1) zero-init identity  mean|Δ|    : {d_identity.mean():.6f}   max|Δ|: {d_identity.max():.6f}")
    print(f"    -> relative to output scale     : {d_identity.mean().item()/max(scale,1e-6):.4%}  (want ~0, small = OK)")
    ratio = (d_effect.mean() / d_identity.mean().clamp_min(1e-9)).item()
    print(f"(2) bias active (λ=2)   mean|Δ|    : {d_effect.mean():.6f}   max|Δ|: {d_effect.max():.6f}")
    print(f"    -> effect / identity-floor ratio: {ratio:.1f}x   mechanism live? {'YES' if ratio > 3 else 'CHECK'}")
    print("    note: identity floor is nonzero because base uses flash-attn and the guided")
    print("          blocks use SDPA (bf16 kernel diff); train+infer both use the guided SDPA")
    print("          path, so the mask ablation stays self-consistent.")
    print("================================\n")


if __name__ == "__main__":
    main()
