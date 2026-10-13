"""
Experiment 3: View Count Ablation
Evaluates ShapeR with 1, 4, 8, and 16 input views, all other settings fixed.

Fixed settings (for fair comparison):
  - 25 denoising steps (same as balance config)
  - token_multiplier = 4 (same as balance/quality config)
  - cluster view selection strategy

The model is loaded once and reused across all view counts to save time.

Usage:
    python experiments/exp3_viewcount_ablation/run_eval.py \
        --input_pkls data/ADT1292__stool.pkl data/BNB2909__pitcher.pkl \
        --num_views 1 4 8 16 \
        --output_dir outputs/exp3_viewcount
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import omegaconf
import torch
import trimesh
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dataset.shaper_dataset import InferenceDataset
from evaluation.eval import _eval_single_mesh, print_metrics, print_summary
from model.flow_matching.shaper_denoiser import ShapeRDenoiser
from model.text.hf_embedder import TextFeatureExtractor
from model.vae3d.autoencoder import MichelangeloLikeAutoencoderWrapper
from postprocessing.helper import remove_floating_geometry


def load_model(device):
    """Load model with balance-equivalent settings (token_multiplier=4, 25 steps)."""
    ckpt_file = "checkpoints/019-0-bfloat16.ckpt"
    state_dict = torch.load(ckpt_file, map_location=device, weights_only=False)
    config = omegaconf.OmegaConf.load("checkpoints/config.yaml")

    print(f"Loading model on {device}...")
    model = ShapeRDenoiser(config).to(device)
    model.convert_to_bfloat16()
    model.load_state_dict(state_dict, strict=False)

    vae = MichelangeloLikeAutoencoderWrapper("checkpoints/vae-088-0-bfloat16.ckpt", device)
    text_fe = TextFeatureExtractor(device=device).to(torch.bfloat16)

    model = torch.compile(model, fullgraph=True)
    model = model.eval()
    vae.model.use_udf_extraction = True
    vae.model.udf_iso = 0.375

    scales = vae.model.get_token_scales()
    scale_prob = np.zeros_like(scales)
    scale_prob[6] = 1.0
    vae.model.set_inference_scale_probabilities(scale_prob)

    token_multiplier = 4  # same as balance/quality config
    token_count = int(scales[np.argmax(scale_prob)].item()) * token_multiplier
    embed_dim = vae.get_embed_dim()
    use_shifted_sampling = getattr(config.fm_transformer, "time_sampler", "lognorm") == "flux"

    return model, vae, text_fe, config, token_count, embed_dim, use_shifted_sampling


def eval_one_condition(pkl_paths, num_views, num_steps, model, vae, text_fe,
                       config, token_count, embed_dim, use_shifted_sampling,
                       device, n_points, output_dir):
    """Run inference and evaluate metrics for a single (num_views, num_steps) condition."""
    dataset = InferenceDataset(config, paths=pkl_paths, override_num_views=num_views)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False,
        num_workers=0, collate_fn=dataset.custom_collate,
    )

    token_shape = (1, token_count, embed_dim)
    results = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"views={num_views}"):
            name = batch["name"][0]
            has_gt = "vertices" in batch and "faces" in batch
            if not has_gt:
                print(f"  [SKIP] {name}: no GT mesh in pkl")
                continue

            batch = InferenceDataset.move_batch_to_device(batch, device, dtype=torch.bfloat16)

            latents = model.infer_latents(
                batch,
                token_shape=token_shape,
                text_feature_extractor=text_fe,
                num_steps=num_steps,
                use_shifted_sampling=use_shifted_sampling,
            )
            pred_mesh = vae.infer_mesh_from_latents(latents)[0]

            try:
                pred_mesh = remove_floating_geometry(pred_mesh)
            except Exception as e:
                print(f"  [WARN] remove_floating_geometry failed: {e}")

            gt_verts = batch["vertices"][0]
            gt_faces = batch["faces"][0]
            if isinstance(gt_verts, torch.Tensor):
                gt_verts = gt_verts.cpu().numpy()
            if isinstance(gt_faces, torch.Tensor):
                gt_faces = gt_faces.cpu().numpy()

            bounds = batch["bounds"][0].float().cpu().numpy()

            metrics = _eval_single_mesh(
                np.array(pred_mesh.vertices, dtype=np.float32),
                np.array(pred_mesh.faces),
                np.array(gt_verts, dtype=np.float32),
                np.array(gt_faces),
                n_points,
                bounds,
            )
            results.append({"name": name, "metrics": metrics})
            print_metrics(name, metrics)

            # Save predicted mesh for qualitative inspection
            cond_dir = Path(output_dir) / f"views_{num_views}"
            cond_dir.mkdir(parents=True, exist_ok=True)
            try:
                tmp = f"/tmp/mesh_{name}_v{num_views}.obj"
                pred_mesh.export(tmp)
                saved_mesh = trimesh.load(tmp, force="mesh")
                saved_mesh.export(cond_dir / f"{name}.glb", include_normals=True)
            except Exception as e:
                print(f"  [WARN] mesh save failed: {e}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkls", nargs="+", required=True,
                        help="Local paths to pkl files.")
    parser.add_argument("--num_views", type=int, nargs="+", default=[1, 4, 8, 16],
                        help="List of view counts to test (default: 1 4 8 16).")
    parser.add_argument("--num_steps", type=int, default=25,
                        help="Denoising steps, fixed for all conditions (default: 25).")
    parser.add_argument("--output_dir", default="outputs/exp3_viewcount")
    parser.add_argument("--n_points", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42). Set -1 to disable.")
    args = parser.parse_args()

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Random seed fixed: {args.seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model (once, shared across all view conditions)...")
    model, vae, text_fe, config, token_count, embed_dim, use_shifted_sampling = load_model(device)

    all_conditions = {}

    for nv in args.num_views:
        print(f"\n{'='*60}")
        print(f"Condition: {nv} view(s), {args.num_steps} denoising steps")
        print(f"{'='*60}")

        results = eval_one_condition(
            args.input_pkls, nv, args.num_steps,
            model, vae, text_fe, config,
            token_count, embed_dim, use_shifted_sampling,
            device, args.n_points, output_dir,
        )

        summary = print_summary(results)
        key = f"views_{nv}"
        all_conditions[key] = {
            "num_views": nv,
            "num_steps": args.num_steps,
            "per_sample": results,
            "summary": summary,
        }

        cond_json = output_dir / f"results_{key}.json"
        with open(cond_json, "w") as f:
            json.dump(all_conditions[key], f, indent=2)
        print(f"Saved {cond_json}")

    # Combined results file
    combined = output_dir / "results_all_views.json"
    with open(combined, "w") as f:
        json.dump(all_conditions, f, indent=2)
    print(f"\nAll results saved to {combined}")

    # Quick comparison table
    print(f"\n{'='*60}")
    print(f"{'Views':>8} | {'F-Score':>10} | {'NC':>10} | {'CD(x1e2)':>12}")
    print(f"{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
    for nv in args.num_views:
        key = f"views_{nv}"
        s = all_conditions[key].get("summary") or {}
        fs = s.get("f-score", {}).get("mean", float("nan"))
        nc = s.get("normals", {}).get("mean", float("nan"))
        cd = s.get("chamfer", {}).get("mean", float("nan")) * 100
        print(f"{nv:>8} | {fs:>10.4f} | {nc:>10.4f} | {cd:>12.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
