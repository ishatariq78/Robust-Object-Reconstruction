# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Run BOTH models on every test condition and save the meshes.

  ORIGINAL  the untouched `ShapeRDenoiser` — the same class ../../infer_shape.py loads.
            No zeroed-parameter stand-in.
  TRAINED   the same checkpoint plus the mask-guided fine-tune (`mask_guided/architecture.py`)
            with its trained lambda and mask conv, fed SAM2 masks exactly as it was trained.

Conditions per object:  clean · occluded · object-standing-on-it

Both models get identical images, identical SAM2 masks and the identical generation seed in
every condition, so a difference between them is the mask guidance and nothing else. The camera
self-test runs first and the job stops if it fails — a wrongly placed object would invalidate
the whole test.

    python updated_shapeR/testing/run_test.py --input_pkls data/BNB1045__wovenstool.pkl
"""

import argparse
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import omegaconf
import torch
import trimesh
from torchsparse import SparseTensor
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]          # HERE is a dir: [0]=updated_shapeR, [1]=repo root
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "updated_shapeR"))
sys.path.insert(0, str(HERE))

from dataset.shaper_dataset import InferenceDataset          # noqa: E402
from model.download import setup_checkpoints                 # noqa: E402
from model.flow_matching.shaper_denoiser import ShapeRDenoiser  # noqa: E402
from model.text.hf_embedder import TextFeatureExtractor      # noqa: E402
from model.vae3d.autoencoder import MichelangeloLikeAutoencoderWrapper  # noqa: E402
from postprocessing.helper import remove_floating_geometry   # noqa: E402
from mask_conditioning import compute_view_foreground_masks  # noqa: E402
from mask_guided.architecture import build_mask_guided_denoiser  # noqa: E402

from augment import occlude, put_on_target, render_through_views, selftest_camera  # noqa: E402

PRESETS = {"quality": (16, 4, 50), "speed": (4, 2, 10), "balance": (8, 4, 25)}


def seed_for(name, cond, base=0):
    return int(zlib.crc32(f"{name}|{cond}|{base}".encode()) & 0x7FFFFFFF)


def fresh_pc(batch):
    pc = batch["semi_dense_points"]
    batch["semi_dense_points"] = SparseTensor(pc.feats, pc.coords, pc.stride)


def sam2_masks(batch, sam2, device):
    b, v, c, h, w = batch["images"].shape
    return compute_view_foreground_masks(
        batch["images"].reshape(b * v, c, h, w).float(),
        batch["masks_ingest"].reshape(b * v, 1, h, w).float(),
        batch["boxes_ingest"].reshape(b * v, 2, 2), sam2, max_points=16,
    ).to(device=device, dtype=torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_pkls", nargs="+", required=True)
    ap.add_argument("--guided_ckpt", default="updated_shapeR/outputs/mask_guided.ckpt")
    ap.add_argument("--config", default="balance", choices=list(PRESETS))
    ap.add_argument("--occ_level", type=float, default=0.6)
    ap.add_argument("--area_ratio", type=float, default=0.25)
    ap.add_argument("--distractor", default="plantpot")
    ap.add_argument("--output_dir", default="updated_shapeR/testing/outputs")
    ap.add_argument("--conditions", nargs="+", default=["clean", "occluded", "onobject"])
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--sam2_model_id", default="facebook/sam2.1-hiera-base-plus")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    n_views, tok_mult, steps = PRESETS[args.config]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    for sub in ("meshes", "views"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    tmp_obj = str(out / f"_tmp_{os.getpid()}.obj")

    setup_checkpoints()
    config = omegaconf.OmegaConf.load("checkpoints/config.yaml")
    ckpt = "checkpoints/019-0-bfloat16.ckpt"

    print("Loading ORIGINAL ShapeRDenoiser (untouched)...")
    original = ShapeRDenoiser(config).to(device)
    original.convert_to_bfloat16()
    original.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    original.eval()

    print("Loading TRAINED mask-guided model...")
    trained, _m, _u = build_mask_guided_denoiser(config, ckpt, device)
    ck = torch.load(args.guided_ckpt, map_location=device, weights_only=False)
    trained.fg_bias_scale.data.copy_(ck["fg_bias_scale"].to(device).view_as(trained.fg_bias_scale))
    if ck.get("mask_conv") is not None:
        trained.dino_ray_extractor._mask_patch_embed[0].proj_in.load_state_dict(
            {k: v.to(device) for k, v in ck["mask_conv"].items()})
    trained.eval()
    print(f"  lambda = {float(trained.fg_bias_scale):.4f}")

    vae = MichelangeloLikeAutoencoderWrapper("checkpoints/vae-088-0-bfloat16.ckpt", device)
    text_fx = TextFeatureExtractor(device=device).to(torch.bfloat16)
    from detect_foreground_sam2 import SAM2ForegroundSegmenter
    sam2 = SAM2ForegroundSegmenter(args.sam2_model_id, device=str(device), dtype=torch.bfloat16)

    vae.model.use_udf_extraction = True
    vae.model.udf_iso = 0.375
    scales = vae.model.get_token_scales()
    sp = np.zeros_like(scales); sp[6] = 1.0
    vae.model.set_inference_scale_probabilities(sp)
    token_shape = (1, int(scales[np.argmax(sp)].item()) * tok_mult, vae.get_embed_dim())
    shifted = getattr(config.fm_transformer, "time_sampler", "lognorm") == "flux"

    ds = InferenceDataset(config, paths=list(args.input_pkls), override_num_views=n_views)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                                         collate_fn=ds.custom_collate)

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="objects"):
            name = batch["name"][0]
            print(f"\n=== {name} ===")
            st = selftest_camera(batch)
            if not st["passed"]:
                raise SystemExit(f"camera self-test failed on {name} — object placement would be wrong")

            gt = trimesh.Trimesh(vertices=np.asarray(batch["vertices"][0], np.float64),
                                 faces=np.asarray(batch["faces"][0]), process=False)
            imgs0 = batch["images"][0].float().cpu().numpy()
            pm0 = batch["masks_ingest"][0].float().cpu().numpy().reshape(-1, 1, *imgs0.shape[-2:])
            ext = batch["camera_extrinsics"][0].cpu().numpy()
            itr = batch["camera_intrinsics"][0].cpu().numpy()
            h, w = imgs0.shape[-2:]
            _rgb, t_alpha, _d = render_through_views(gt, ext, itr, h, w)

            variants = {"clean": {"images": imgs0, "point_masks": pm0, "boxes": None, "stats": {}}}
            if "occluded" in args.conditions:
                variants["occluded"] = occlude(imgs0, pm0, t_alpha, args.occ_level,
                                               seed=seed_for(name, "occ", args.seed_base))
                print(f"  occlusion: {variants['occluded']['stats']['achieved']*100:.0f}% of the "
                      f"object hidden, {variants['occluded']['stats']['points_kept']*100:.0f}% of dots kept")
            if "onobject" in args.conditions:
                r = put_on_target(imgs0, gt, ext, itr, args.distractor, args.area_ratio)
                variants["onobject"] = {"images": r["images"], "point_masks": pm0,
                                        "boxes": None, "stats": r["stats"], "drawn": r["drawn"]}

            batch = InferenceDataset.move_batch_to_device(batch, device, dtype=torch.bfloat16)
            base_i, base_m, base_b = (batch["images"].clone(), batch["masks_ingest"].clone(),
                                      batch["boxes_ingest"].clone())
            rec = manifest.get(name, {})
            rec["camera_selftest_px"] = st["median_px"]

            for cond in args.conditions:
                v = variants[cond]
                batch["images"] = torch.from_numpy(v["images"]).unsqueeze(0).to(device, base_i.dtype)
                batch["masks_ingest"] = torch.from_numpy(v["point_masks"]).reshape(
                    base_m.shape).to(device, base_m.dtype)
                batch["boxes_ingest"] = (base_b if v["boxes"] is None else
                                         torch.from_numpy(v["boxes"]).reshape(base_b.shape)
                                         .to(device, base_b.dtype))
                fg = sam2_masks(batch, sam2, device)
                np.save(out / "views" / f"{name}__{cond}__images.npy",
                        v["images"].astype(np.float32))
                np.save(out / "views" / f"{name}__{cond}__sam2.npy",
                        fg[:, 0].float().cpu().numpy().astype(np.float32))
                rec.setdefault("stats", {})[cond] = v["stats"]

                for model_name, model in (("original", original), ("trained", trained)):
                    dst = out / "meshes" / f"{name}__{cond}__{model_name}.glb"
                    rec.setdefault("meshes", {})[f"{cond}__{model_name}"] = str(dst)
                    if args.skip_existing and dst.exists():
                        print(f"    skip {dst.name}"); continue
                    if model_name == "trained":
                        model._fg_masks = fg
                    try:
                        s = seed_for(name, cond, args.seed_base)   # shared by both models
                        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
                        fresh_pc(batch)
                        lat = model.infer_latents(batch, token_shape=token_shape,
                                                  text_feature_extractor=text_fx,
                                                  num_steps=steps, use_shifted_sampling=shifted)
                        mesh = vae.infer_mesh_from_latents(lat)[0]
                        mesh = remove_floating_geometry(mesh)
                        mesh = mesh.simplify_quadric_decimation(face_count=125000)
                        mesh = ds.rescale_back(batch["index"][0], mesh, False)
                        mesh.export(tmp_obj)
                        trimesh.load(tmp_obj, force="mesh").export(dst, include_normals=True)
                        print(f"    saved {dst.name}")
                    except Exception as e:
                        print(f"    FAILED {dst.name}: {type(e).__name__}: {e}")
                    finally:
                        model._fg_masks = None if model_name == "trained" else None

            batch["images"], batch["masks_ingest"], batch["boxes_ingest"] = base_i, base_m, base_b
            manifest[name] = rec                    # merge, never overwrite other objects
            manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nDone. meshes: {out/'meshes'}   views: {out/'views'}   manifest: {manifest_path}")


if __name__ == "__main__":
    main()
