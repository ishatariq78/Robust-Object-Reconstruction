# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
ShapeR inference WITH SAM2 foreground-mask conditioning.

Same as infer_shape.py, but before generating the mesh it masks the background out of
every DINO input view (using SAM2 foreground masks computed on ShapeR's own cropped
views). DINO then never sees background clutter, so the reconstruction does not grow
extra background objects. See mask_conditioning.py for the mechanism.

Usage:
    python infer_shape_masked.py --input_pkl ADT1292__stool.pkl --config balance --is_local_path \
        --save_masked_views --output_dir output_masked

    # A/B against the unconditioned baseline:
    python infer_shape.py         --input_pkl ADT1292__stool.pkl --config balance --output_dir output_baseline
    python infer_shape_masked.py  --input_pkl ADT1292__stool.pkl --config balance --output_dir output_masked

Notes:
  - torch.compile is intentionally disabled here: the background-masking hook is
    data-dependent and a fullgraph-compiled model would not see it.
  - Only the DINO image branch is changed. Point cloud, text, camera rays and the
    existing point-projection mask conditioning are untouched.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import omegaconf
import torch
import trimesh
from tqdm import tqdm

from dataset.download import setup_data
from dataset.shaper_dataset import InferenceDataset
from mask_conditioning import (
    compute_view_foreground_masks,
    install_mask_conditioning,
    make_masked_view_grid,
)
from model.download import setup_checkpoints
from model.flow_matching.shaper_denoiser import ShapeRDenoiser
from model.text.hf_embedder import TextFeatureExtractor
from model.vae3d.autoencoder import MichelangeloLikeAutoencoderWrapper
from postprocessing.helper import remove_floating_geometry

# Preset configs: (num_images, token_multiplier, num_denoising_steps)
preset_configs = {
    "quality": (16, 4, 50),
    "speed": (4, 2, 10),
    "balance": (16, 4, 25),
}


def build_sam2_segmenter(model_id, device):
    """Import and construct the SAM2 segmenter from the updated_shapeR folder."""
    sys.path.insert(0, str(Path(__file__).parent / "updated_shapeR"))
    from detect_foreground_sam2 import SAM2ForegroundSegmenter

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    return SAM2ForegroundSegmenter(model_id, device=str(device), dtype=dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, default="ADT1292__stool.pkl")
    parser.add_argument("--remove_floating_geometry", action="store_false")
    parser.add_argument("--simplify_mesh", action="store_false")
    parser.add_argument("--output_dir", type=str, default="output_masked")
    parser.add_argument("--config", type=str, default="balance")
    parser.add_argument("--do_transform_to_world", action="store_true")
    parser.add_argument("--is_local_path", action="store_true")
    # mask-conditioning specific
    parser.add_argument("--sam2_model_id", type=str, default="facebook/sam2.1-hiera-base-plus")
    parser.add_argument("--neutral_value", type=float, default=0.5,
                        help="Gray value (in [0,1]) that replaces background pixels before DINO.")
    parser.add_argument("--max_prompt_points", type=int, default=16,
                        help="Max SLAM point clicks used to prompt SAM2 per view.")
    parser.add_argument("--save_masked_views", action="store_true",
                        help="Save a grid of the background-masked views actually fed to DINO.")
    parser.add_argument("--disable_mask", action="store_true",
                        help="Skip mask conditioning (behaves like plain infer_shape.py) for A/B testing.")
    args = parser.parse_args()

    num_images, token_multiplier, num_steps = preset_configs[args.config]

    setup_checkpoints()
    if not args.is_local_path:
        setup_data(args.input_pkl)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load("checkpoints/019-0-bfloat16.ckpt", map_location=device, weights_only=False)
    config = omegaconf.OmegaConf.load("checkpoints/config.yaml")

    print("Loading model...")
    model = ShapeRDenoiser(config).to(device)
    model.convert_to_bfloat16()
    model.load_state_dict(state_dict, strict=False)
    model = model.eval()  # NOTE: no torch.compile -- the masking hook is data-dependent

    vae = MichelangeloLikeAutoencoderWrapper("checkpoints/vae-088-0-bfloat16.ckpt", device)
    text_feature_extractor = TextFeatureExtractor(device=device).to(torch.bfloat16)

    vae.model.use_udf_extraction = True
    vae.model.udf_iso = 0.375
    scales = vae.model.get_token_scales()
    scale_prob = np.zeros_like(scales)
    scale_prob[6] = 1.0
    vae.model.set_inference_scale_probabilities(scale_prob)
    token_count = int(scales[np.argmax(scale_prob)].item()) * token_multiplier
    token_shape = (1, token_count, vae.get_embed_dim())
    use_shifted_sampling = getattr(config.fm_transformer, "time_sampler", "lognorm") == "flux"

    sam2 = None if args.disable_mask else build_sam2_segmenter(args.sam2_model_id, device)

    sample_path = os.path.join("data", args.input_pkl) if not args.is_local_path else args.input_pkl
    print("Loading input pkl from", args.input_pkl)
    inference_dataset = InferenceDataset(config, paths=[sample_path], override_num_views=num_images)
    inference_loader = torch.utils.data.DataLoader(
        inference_dataset, batch_size=1, shuffle=False, drop_last=False,
        num_workers=0, collate_fn=inference_dataset.custom_collate,
    )

    with torch.no_grad():
        for batch in tqdm(inference_loader):
            batch = InferenceDataset.move_batch_to_device(batch, device, dtype=torch.bfloat16)

            # -----------------------------------------------------------------
            # Mask conditioning: SAM2 foreground per view -> background masking
            # -----------------------------------------------------------------
            if not args.disable_mask:
                b, v, c, h, w = batch["images"].shape
                images_bv = batch["images"].reshape(b * v, c, h, w)
                point_masks_bv = batch["masks_ingest"].reshape(b * v, 1, h, w)
                boxes_bv = batch["boxes_ingest"].reshape(b * v, 2, 2)

                print(f"Running SAM2 on {b * v} views for foreground masks...")
                fg_masks = compute_view_foreground_masks(
                    images_bv.float(), point_masks_bv.float(), boxes_bv,
                    sam2, max_points=args.max_prompt_points,
                )
                cover = fg_masks.mean().item()
                print(f"  mean foreground coverage across views: {cover:.3f}")

                if args.save_masked_views:
                    grid = make_masked_view_grid(images_bv.float(), fg_masks, args.neutral_value)
                    grid_path = output_dir / f"MASKEDVIEWS__{batch['name'][0]}.jpg"
                    grid.save(grid_path)
                    print(f"  saved masked-view grid to {grid_path}")

                install_background_masking(
                    model.dino_ray_extractor,
                    fg_masks.to(device=device, dtype=torch.bfloat16),
                    neutral_value=args.neutral_value,
                )

            # -----------------------------------------------------------------
            # Standard ShapeR generation (now conditioned on masked views)
            # -----------------------------------------------------------------
            latents_pred = model.infer_latents(
                batch,
                token_shape=token_shape,
                text_feature_extractor=text_feature_extractor,
                num_steps=num_steps,
                use_shifted_sampling=use_shifted_sampling,
            )
            mesh = vae.infer_mesh_from_latents(latents_pred)[0]

            if args.remove_floating_geometry:
                mesh = remove_floating_geometry(mesh)
            if args.simplify_mesh:
                mesh = mesh.simplify_quadric_decimation(face_count=125000)
            mesh = inference_dataset.rescale_back(batch["index"][0], mesh, args.do_transform_to_world)

            tmp_output_path_mesh = "/tmp/mesh_masked.obj"
            mesh.export(tmp_output_path_mesh)
            mesh = trimesh.load(tmp_output_path_mesh, force="mesh")
            out_path = output_dir / (batch["name"][0] + ".glb")
            mesh.export(out_path, include_normals=True)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
