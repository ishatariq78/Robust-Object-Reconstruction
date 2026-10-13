# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Three arms on the same cluttered inputs, same seeds, so only the mask differs.

  control  guidance switched off (fg_bias_scale=0, zero-init mask conv) — the same code path
           with the fine-tune inert. This is the "before".
  stock    the fine-tune with masks exactly as the pipeline builds them today: SAM2 prompted
           with points sampled from the SLAM projection mask, which in a cluttered scene
           INCLUDES the intruder's points.
  fixed    the fine-tune with masks prompted target-points-positive / intruder-points-negative,
           after splitting the cloud with the support segmenter. No weights change.

Every arm shares the object, the clutter, the views and the generation seed, so a difference
between them is the mask and nothing else.

Also saves, per view, the input image with both masks overlaid — the mask is the thing being
changed, so it has to be visible, not just asserted.

    python updated_2/mask_vs_leak/three_arms.py --objects <pkl> ... --seeds 0 1 2
"""

import argparse
import json
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

from clean_masks import masks_with_negatives                     # noqa: E402
from segment_v2 import segment_by_support_v2                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", required=True)
    ap.add_argument("--mask_pkls", nargs="+", default=None,
                    help="parallel list: build the MASK from these clouds while GENERATING from "
                         "--objects. Needed for the cleaned run: once the intruder's points are "
                         "removed the segmenter finds nothing, so the fixed mask would collapse "
                         "back to the stock one. Cameras and images are identical between the "
                         "two files, only the points differ, so the projection stays valid.")
    ap.add_argument("--arms", nargs="+", default=["control", "stock", "fixed"],
                    choices=["control", "stock", "fixed"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out_dir", default=str(HERE / "outputs" / "three_arms"))
    ap.add_argument("--guided_ckpt", default=str(REPO / "updated_shapeR/outputs/mask_guided.ckpt"))
    ap.add_argument("--sam2_model_id", default="facebook/sam2.1-hiera-base-plus")
    args = ap.parse_args()

    import omegaconf
    from dataset.shaper_dataset import InferenceDataset
    from model.download import setup_checkpoints
    from model.text.hf_embedder import TextFeatureExtractor
    from model.vae3d.autoencoder import MichelangeloLikeAutoencoderWrapper
    from postprocessing.helper import remove_floating_geometry
    from mask_conditioning import compute_view_foreground_masks
    from mask_guided.architecture import build_mask_guided_denoiser
    from detect_foreground_sam2 import SAM2ForegroundSegmenter
    from torchsparse import SparseTensor

    out = Path(args.out_dir)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    setup_checkpoints()
    config = omegaconf.OmegaConf.load(str(REPO / "checkpoints/config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = str(REPO / "checkpoints/019-0-bfloat16.ckpt")

    model, _m, _u = build_mask_guided_denoiser(config, ckpt, device)
    trained = torch.load(args.guided_ckpt, map_location=device, weights_only=False)
    model.eval()

    vae = MichelangeloLikeAutoencoderWrapper(str(REPO / "checkpoints/vae-088-0-bfloat16.ckpt"), device)
    text_fx = TextFeatureExtractor(device=device).to(torch.bfloat16)
    sam2 = SAM2ForegroundSegmenter(args.sam2_model_id, device=str(device), dtype=torch.bfloat16)

    vae.model.use_udf_extraction = True
    vae.model.udf_iso = 0.375
    scales = vae.model.get_token_scales()
    sp = np.zeros_like(scales); sp[6] = 1.0
    vae.model.set_inference_scale_probabilities(sp)
    token_shape = (1, int(scales[np.argmax(sp)].item()) * 4, vae.get_embed_dim())
    shifted = getattr(config.fm_transformer, "time_sampler", "lognorm") == "flux"

    def set_guidance(on):
        if on:
            model.fg_bias_scale.data.copy_(
                trained["fg_bias_scale"].to(device).view_as(model.fg_bias_scale))
            if trained.get("mask_conv") is not None:
                model.dino_ray_extractor._mask_patch_embed[0].proj_in.load_state_dict(
                    {k: v.to(device) for k, v in trained["mask_conv"].items()})
        else:
            model.fg_bias_scale.data.zero_()
            for p in model.dino_ray_extractor._mask_patch_embed[0].proj_in.parameters():
                p.data.zero_()

    manifest = {}
    mpath = out / "manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text())

    ds = InferenceDataset(config, paths=list(args.objects), override_num_views=8)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                                         collate_fn=ds.custom_collate)

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            name = batch["name"][0]
            pkl = args.objects[bi]
            print(f"\n=== {name} ===", flush=True)
            batch = InferenceDataset.move_batch_to_device(batch, device, dtype=torch.bfloat16)
            b, v, c, h, w = batch["images"].shape

            mask_src = (args.mask_pkls[bi] if args.mask_pkls else pkl)
            d = pickle.load(open(mask_src, "rb"))
            cloud = d["points_model"].numpy().astype(np.float64)[:, :3]
            scale = 0.9 / float(np.max(d["bounds"].numpy()))
            is_target = segment_by_support_v2(cloud)
            print(f"  segmenter called {int((~is_target).sum())}/{len(cloud)} points intruder")

            fg_stock = compute_view_foreground_masks(
                batch["images"].reshape(b * v, c, h, w).float(),
                batch["masks_ingest"].reshape(b * v, 1, h, w).float(),
                batch["boxes_ingest"].reshape(b * v, 2, 2), sam2, max_points=16
            ).to(device=device, dtype=torch.bfloat16)
            fg_fixed, _stats = masks_with_negatives(batch, cloud * scale, is_target, sam2)
            fg_fixed = fg_fixed.to(device=device, dtype=torch.bfloat16)

            a_s = float(fg_stock.float().mean()); a_f = float(fg_fixed.float().mean())
            print(f"  mask foreground area  stock {a_s:.3f} -> fixed {a_f:.3f} "
                  f"({(a_f-a_s)/max(a_s,1e-9)*100:+.0f}%)")
            np.savez_compressed(out / "masks" / f"{name}.npz",
                                images=batch["images"][0].float().cpu().numpy(),
                                stock=fg_stock.float().cpu().numpy(),
                                fixed=fg_fixed.float().cpu().numpy())

            rec = manifest.get(name, {})
            for arm, guid, fg in [(a, g, f) for a, g, f in
                                  (("control", False, fg_stock),
                                   ("stock", True, fg_stock),
                                   ("fixed", True, fg_fixed)) if a in args.arms]:
                set_guidance(guid)
                model._fg_masks = fg
                for s in args.seeds:
                    dst = out / "meshes" / f"{name}__{arm}__s{s}.glb"
                    rec[f"{arm}__s{s}"] = str(dst)
                    if dst.exists():
                        continue
                    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
                    pc = batch["semi_dense_points"]
                    batch["semi_dense_points"] = SparseTensor(pc.feats, pc.coords, pc.stride)
                    try:
                        lat = model.infer_latents(batch, token_shape=token_shape,
                                                  text_feature_extractor=text_fx,
                                                  num_steps=25, use_shifted_sampling=shifted)
                        mesh = vae.infer_mesh_from_latents(lat)[0]
                        mesh = remove_floating_geometry(mesh)
                        mesh = mesh.simplify_quadric_decimation(face_count=125000)
                        mesh = ds.rescale_back(batch["index"][0], mesh, False)
                        tmp = str(out / f"_t{s}.obj")
                        mesh.export(tmp)
                        trimesh.load(tmp, force="mesh").export(dst, include_normals=True)
                    except Exception as e:
                        print(f"    FAILED {dst.name}: {type(e).__name__}: {e}")
                model._fg_masks = None
                print(f"  {arm}: done")
            manifest[name] = rec
            mpath.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
