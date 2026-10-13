# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
The decisive test of the mask channel: give the model a PERFECT mask.

THE QUESTION. Every mask experiment so far used a mask SAM2 guessed. When it failed to suppress
the intruder we could never separate two explanations: the mask channel is too weak to matter, or
the mask was simply wrong. A perfect mask separates them.

THE SETUP. `image`-mode clutter: the pot is painted into the photos but is NOT in the scan
points. So the point branch has no knowledge of it whatsoever, and the ONLY place the model can
learn the pot exists is the images. That is the single condition where mask guidance could
possibly decide the outcome — and the one condition where the fine-tune has already shown a real
effect (Leak% -0.55).

Three arms on identical inputs and seeds:
  off      guidance disabled
  sam2     the fine-tune with the mask the pipeline builds today
  oracle   the fine-tune with the TARGET'S TRUE SILHOUETTE, rendered from its ground-truth mesh
           through the same cameras. The pot is excluded by construction, not by a segmenter.

The oracle mask is not deployable — it needs the answer. It is an upper bound, and that is the
point: if a perfect mask cannot suppress the pot, no better segmentation ever will, and effort
on mask quality is wasted. If it can, mask quality is the whole game.

    python updated_2/mask_vs_leak/oracle_mask.py --objects <clutter pkl> ...
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
sys.path.insert(0, str(REPO / "updated_shapeR/testing"))


def oracle_masks(batch, gt_mesh_scaled, occluder=None):
    """
    Pixels where the TARGET is the visible surface — what a flawless segmenter would produce.

    THE BUG THIS FIXES, found by checking the mask instead of assuming it. The first version
    returned the target's raw silhouette. But the intruder stands IN FRONT of the target, so the
    target's silhouette covers the pixels the intruder occludes: measured against ground truth,
    that "oracle" mask covered 76-100% of the pot. It was telling the model "target" exactly
    where the pot is — the opposite of its purpose, and it would have made the upper-bound
    experiment meaningless.

    With the occluder supplied, pixels where the occluder is NEARER the camera are removed by a
    real depth test, so the mask is the target's VISIBLE surface. Without one it falls back to
    the silhouette, which is correct only when nothing stands in front.
    """
    from augment import render_through_views
    imgs = batch["images"]
    b, v, c, h, w = imgs.shape
    ext = batch["camera_extrinsics"][0].float().cpu().numpy()
    itr = batch["camera_intrinsics"][0].float().cpu().numpy()
    _rgb, alpha_t, depth_t = render_through_views(gt_mesh_scaled, ext, itr, h, w)
    vis = alpha_t > 0.5
    if occluder is not None:
        _r2, alpha_o, depth_o = render_through_views(occluder, ext, itr, h, w)
        hit_o = alpha_o > 0.5
        # depth 0 means "nothing hit"; only compare where both surfaces exist
        nearer = hit_o & (depth_o > 0) & ((depth_t <= 0) | (depth_o < depth_t))
        vis = vis & ~nearer
    m = torch.as_tensor(vis.astype(np.float32), device=imgs.device)
    return m.reshape(b * v, 1, h, w).to(imgs.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out_dir", default=str(HERE / "outputs" / "oracle"))
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
    setup_checkpoints()
    config = omegaconf.OmegaConf.load(str(REPO / "checkpoints/config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _m, _u = build_mask_guided_denoiser(
        config, str(REPO / "checkpoints/019-0-bfloat16.ckpt"), device)
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

    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    ds = InferenceDataset(config, paths=list(args.objects), override_num_views=8)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                                         collate_fn=ds.custom_collate)

    with torch.no_grad():
        for _bi, batch in enumerate(loader):
            name = batch["name"][0]
            print(f"\n=== {name} ===", flush=True)
            gt = trimesh.Trimesh(vertices=np.asarray(batch["vertices"][0], np.float64),
                                 faces=np.asarray(batch["faces"][0]), process=False)
            batch = InferenceDataset.move_batch_to_device(batch, device, dtype=torch.bfloat16)
            b, v, c, h, w = batch["images"].shape
            fg_sam = compute_view_foreground_masks(
                batch["images"].reshape(b * v, c, h, w).float(),
                batch["masks_ingest"].reshape(b * v, 1, h, w).float(),
                batch["boxes_ingest"].reshape(b * v, 2, 2), sam2, max_points=16
            ).to(device=device, dtype=torch.bfloat16)
            # The occluder MUST be passed. Without it the "oracle" is the target's raw
            # silhouette, which covers the pot standing in front of it — measured at 76-100%
            # pot coverage before this was caught, i.e. the exact opposite of the intended mask.
            _lab = Path(args.objects[_bi]).with_name(Path(args.objects[_bi]).stem + "__label.npz")
            _z = np.load(_lab)
            _d = pickle.load(open(args.objects[_bi], "rb"))
            _sc = 0.9 / float(np.max(_d["bounds"].numpy()))
            _pot = trimesh.Trimesh(vertices=_z["distractor_vertices"] * _sc,
                                   faces=_z["distractor_faces"], process=False)
            fg_or = oracle_masks(batch, gt, occluder=_pot).to(device=device, dtype=torch.bfloat16)
            a_s, a_o = float(fg_sam.float().mean()), float(fg_or.float().mean())
            print(f"  mask area  SAM2 {a_s:.3f}   oracle {a_o:.3f}   "
                  f"SAM2 is {(a_s-a_o)/max(a_o,1e-9)*100:+.0f}% vs the true silhouette")
            rec = manifest.get(name, {})
            for arm, guid, fg in (("off", False, fg_sam), ("sam2", True, fg_sam),
                                  ("oracle", True, fg_or)):
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
