# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Inference / testing with the mask-guided ShapeR model.

Loads pretrained ShapeR + the trained mask-guidance weights (lambda + widened mask conv),
computes SAM2 foreground masks per view, and reconstructs meshes with Option A/B active.

Run from the ShapeR repo root:

    python updated_shapeR/mask_guided/infer.py \
        --input_pkls data/BNB1045__sofa.pkl \
        --guided_ckpt updated_shapeR/outputs/mask_guided.ckpt \
        --config balance --output_dir updated_shapeR/outputs/recon_mask_guided
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import omegaconf
import torch
import trimesh
from torchsparse import SparseTensor
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "updated_shapeR"))

from dataset.shaper_dataset import InferenceDataset
from model.download import setup_checkpoints
from model.text.hf_embedder import TextFeatureExtractor
from model.vae3d.autoencoder import MichelangeloLikeAutoencoderWrapper
from postprocessing.helper import remove_floating_geometry

from mask_conditioning import compute_view_foreground_masks
from mask_guided.architecture import build_mask_guided_denoiser

preset_configs = {"quality": (16, 4, 50), "speed": (4, 2, 10), "balance": (8, 4, 25)}  # note: balance=8 views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_pkls", nargs="+", required=True)
    ap.add_argument("--guided_ckpt", required=True, help="Checkpoint from train.py (lambda + mask conv).")
    ap.add_argument("--config", default="balance")
    ap.add_argument("--output_dir", default="updated_shapeR/outputs/recon_mask_guided")
    ap.add_argument("--sam2_model_id", default="facebook/sam2.1-hiera-base-plus")
    ap.add_argument("--do_transform_to_world", action="store_true")
    ap.add_argument("--disable_guidance", action="store_true",
                    help="Do NOT load the trained weights (lambda=0, mask conv stays zero-init) — a clean "
                         "SDPA-path BASELINE for fair A/B against the mask-guided run.")
    ap.add_argument("--num_seeds", type=int, default=1,
                    help="Reconstruct each sample this many times with different seeds (averages out the "
                         "stochastic generation). N>1 saves <name>__s{k}.glb.")
    ap.add_argument("--seed_base", type=int, default=0)
    args = ap.parse_args()

    num_images, token_multiplier, num_steps = preset_configs[args.config]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_checkpoints()
    config = omegaconf.OmegaConf.load("checkpoints/config.yaml")

    model, _, _ = build_mask_guided_denoiser(config, "checkpoints/019-0-bfloat16.ckpt", device)
    if args.disable_guidance:
        print("BASELINE mode: guidance disabled (fg_bias_scale=0, mask conv zero-init) — same SDPA path.")
    else:
        ck = torch.load(args.guided_ckpt, map_location=device, weights_only=False)
        model.fg_bias_scale.data.copy_(ck["fg_bias_scale"].to(device).view_as(model.fg_bias_scale))
        if ck.get("mask_conv") is not None:
            model.dino_ray_extractor._mask_patch_embed[0].proj_in.load_state_dict(ck["mask_conv"])
        print(f"Loaded mask-guidance weights: fg_bias_scale={float(model.fg_bias_scale):.4f}")
    model.eval()

    vae = MichelangeloLikeAutoencoderWrapper("checkpoints/vae-088-0-bfloat16.ckpt", device)
    text_fx = TextFeatureExtractor(device=device).to(torch.bfloat16)
    from detect_foreground_sam2 import SAM2ForegroundSegmenter
    sam2 = SAM2ForegroundSegmenter(args.sam2_model_id, device=str(device), dtype=torch.bfloat16)

    vae.model.use_udf_extraction = True
    vae.model.udf_iso = 0.375
    scales = vae.model.get_token_scales()
    scale_prob = np.zeros_like(scales); scale_prob[6] = 1.0
    vae.model.set_inference_scale_probabilities(scale_prob)
    token_count = int(scales[np.argmax(scale_prob)].item()) * token_multiplier
    token_shape = (1, token_count, vae.get_embed_dim())
    use_shifted = getattr(config.fm_transformer, "time_sampler", "lognorm") == "flux"

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    ds = InferenceDataset(config, paths=list(args.input_pkls), override_num_views=num_images)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=ds.custom_collate)

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = InferenceDataset.move_batch_to_device(batch, device, dtype=torch.bfloat16)
            b, v, c, h, w = batch["images"].shape
            fg = compute_view_foreground_masks(
                batch["images"].reshape(b * v, c, h, w).float(),
                batch["masks_ingest"].reshape(b * v, 1, h, w).float(),
                batch["boxes_ingest"].reshape(b * v, 2, 2), sam2, max_points=16,
            ).to(device=device, dtype=torch.bfloat16)
            model._fg_masks = fg
            name = batch["name"][0]

            for k in range(args.num_seeds):
                seed = args.seed_base + k
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                latents = model.infer_latents(batch, token_shape=token_shape, text_feature_extractor=text_fx,
                                              num_steps=num_steps, use_shifted_sampling=use_shifted)
                mesh = vae.infer_mesh_from_latents(latents)[0]
                mesh = remove_floating_geometry(mesh)
                mesh = mesh.simplify_quadric_decimation(face_count=125000)
                mesh = ds.rescale_back(batch["index"][0], mesh, args.do_transform_to_world)
                suffix = f"__s{k}" if args.num_seeds > 1 else ""
                mesh.export("/tmp/mg_mesh.obj")
                out_path = out / (name + suffix + ".glb")
                trimesh.load("/tmp/mg_mesh.obj", force="mesh").export(out_path, include_normals=True)
                print(f"Saved {out_path}")
            model._fg_masks = None


if __name__ == "__main__":
    main()
