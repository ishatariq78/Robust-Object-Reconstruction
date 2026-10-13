# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Stage 1 — demonstrate the limitation.

Runs the ORIGINAL, untouched ShapeR on the clean and cluttered versions of the
same object and measures how much of the intruding object ends up in the mesh.

This is a controlled A/B: `clutter.py` builds both conditions from the SAME pruned
view pool, so ShapeR's own view selection sees identical candidates, and the only
difference is the clutter. Every condition runs at several seeds because ShapeR's
inference is NOT deterministic — `get_x0_from_input` draws the initial latent from
the global RNG and `infer_shape.py` never seeds it. The measured run-to-run Chamfer
spread is roughly a quarter of the mean, which is larger than most effects worth
reporting, so a single-seed number here would be meaningless. The seed is set
explicitly per cell and shared across conditions.

The headline number is Leak% (see `metrics.py`): predicted surface that sits on the
injected distractor rather than the target. Unlike Extra%, it cannot be inflated by
ordinary fidelity error.

    python updated_2/demo_limitation.py \
        --input_pkls data/ADT1292__woodentable.pkl \
        --levels 0.15 0.25 0.40 --modes image points both --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import omegaconf
import torch
import trimesh
from torchsparse import SparseTensor

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from dataset.shaper_dataset import InferenceDataset            # noqa: E402
from model.download import setup_checkpoints                   # noqa: E402
from model.flow_matching.shaper_denoiser import ShapeRDenoiser  # noqa: E402
from model.text.hf_embedder import TextFeatureExtractor        # noqa: E402
from model.vae3d.autoencoder import MichelangeloLikeAutoencoderWrapper  # noqa: E402
from postprocessing.helper import remove_floating_geometry     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clutter as C                                            # noqa: E402
import metrics as M                                            # noqa: E402

PRESETS = {"quality": (16, 4, 50), "speed": (4, 2, 10), "balance": (8, 4, 25)}


def fresh_pc(batch):
    """`infer_latents` consumes the SparseTensor; rebuild it before each run."""
    pc = batch["semi_dense_points"]
    batch["semi_dense_points"] = SparseTensor(pc.feats, pc.coords, pc.stride)


class Runner:
    """Loads the original model once and reconstructs from any pkl path."""

    def __init__(self, config_name="balance", device=None):
        self.n_views, self.tok_mult, self.steps = PRESETS[config_name]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        setup_checkpoints()
        self.config = omegaconf.OmegaConf.load("checkpoints/config.yaml")
        ckpt = "checkpoints/019-0-bfloat16.ckpt"

        print("Loading ORIGINAL ShapeRDenoiser (untouched)...")
        self.model = ShapeRDenoiser(self.config).to(self.device)
        self.model.convert_to_bfloat16()
        self.model.load_state_dict(
            torch.load(ckpt, map_location=self.device, weights_only=False))
        self.model.eval()

        self.vae = MichelangeloLikeAutoencoderWrapper(
            "checkpoints/vae-088-0-bfloat16.ckpt", self.device)
        self.text_fx = TextFeatureExtractor(device=self.device).to(torch.bfloat16)

        self.vae.model.use_udf_extraction = True
        self.vae.model.udf_iso = 0.375
        scales = self.vae.model.get_token_scales()
        sp = np.zeros_like(scales)
        sp[6] = 1.0
        self.vae.model.set_inference_scale_probabilities(sp)
        self.token_shape = (1, int(scales[np.argmax(sp)].item()) * self.tok_mult,
                            self.vae.get_embed_dim())
        self.shifted = getattr(self.config.fm_transformer, "time_sampler", "lognorm") == "flux"

    def batch_for(self, pkl_path):
        ds = InferenceDataset(self.config, paths=[str(pkl_path)],
                              override_num_views=self.n_views)
        loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                             num_workers=0, collate_fn=ds.custom_collate)
        batch = next(iter(loader))
        return ds, InferenceDataset.move_batch_to_device(
            batch, self.device, dtype=torch.bfloat16)

    @torch.no_grad()
    def reconstruct(self, pkl_path, seed, out_glb=None, return_latents=False):
        ds, batch = self.batch_for(pkl_path)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        fresh_pc(batch)
        lat = self.model.infer_latents(
            batch, token_shape=self.token_shape,
            text_feature_extractor=self.text_fx,
            num_steps=self.steps, use_shifted_sampling=self.shifted)
        mesh = self.vae.infer_mesh_from_latents(lat)[0]
        mesh = remove_floating_geometry(mesh)
        mesh = mesh.simplify_quadric_decimation(face_count=125000)
        mesh = ds.rescale_back(batch["index"][0], mesh, False)
        if out_glb is not None:
            Path(out_glb).parent.mkdir(parents=True, exist_ok=True)
            tmp = str(Path(out_glb).with_suffix(".tmp.obj"))
            mesh.export(tmp)
            trimesh.load(tmp, force="mesh").export(out_glb, include_normals=True)
            os.remove(tmp)
        return (mesh, lat) if return_latents else mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_pkls", nargs="+", required=True)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.25])
    ap.add_argument("--modes", nargs="+", default=["image", "points", "both"],
                    choices=["image", "points", "both"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--distractor", default="plantpot")
    ap.add_argument("--n_views_pool", type=int, default=48)
    ap.add_argument("--config", default="balance", choices=list(PRESETS))
    ap.add_argument("--out_dir", default="updated_2/outputs/stage1")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    clutter_dir = out / "clutter"
    mesh_dir = out / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # ---- build every cluttered variant first (CPU only) ----
    built = {}
    for pkl in args.input_pkls:
        for mode in args.modes:
            for lv in args.levels:
                b = C.build(pkl, clutter_dir, lv, mode, args.distractor,
                            args.n_views_pool, "slam", seed=0)
                built[(Path(pkl).stem, mode, lv)] = b

    runner = Runner(args.config)
    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done = {(r["name"], r["mode"], r["level"], r["condition"], r["seed"]) for r in results}

    # The clean condition depends only on (object, seed) — the clean pkl is byte
    # identical across modes and levels — so it is reconstructed ONCE and the same
    # mesh is scored into every cell. Re-running it per cell would burn most of the
    # GPU budget regenerating the same shape, and would also make the pairing
    # noisier rather than tighter, since each rerun draws a different latent.
    clean_cache = {}

    for (name, mode, lv), b in built.items():
        tgt, dis = M.load_label(b["label"])
        # The clean condition is scored against the distractor too. That is not a
        # mistake: the intruder stands ON the target, so genuine target surface
        # falls within tolerance of it and registers as "leak". Scoring the clean
        # mesh against the same label measures that false-positive floor, and the
        # quantity to report is the RISE in Leak% over it, never the raw value.
        for cond, pkl_path, label in (("clean", b["clean_pkl"], dis),
                                      ("clutter", b["clutter_pkl"], dis)):
            for seed in args.seeds:
                if (name, mode, lv, cond, seed) in done:
                    continue
                if cond == "clean":
                    tag = f"{name}__clean__s{seed}"
                    if (name, seed) in clean_cache:
                        mesh = clean_cache[(name, seed)]
                    else:
                        glb = mesh_dir / f"{tag}.glb"
                        if glb.exists() and args.skip_existing:
                            mesh = trimesh.load(glb, force="mesh")
                        else:
                            print(f"  [{tag}]")
                            try:
                                mesh = runner.reconstruct(pkl_path, seed, out_glb=glb)
                            except Exception as e:
                                print(f"    FAILED: {type(e).__name__}: {e}")
                                continue
                        clean_cache[(name, seed)] = mesh
                    glb = mesh_dir / f"{tag}.glb"
                else:
                    tag = f"{name}__{mode}__L{int(round(lv*100)):03d}__clutter__s{seed}"
                    glb = mesh_dir / f"{tag}.glb"
                    if glb.exists() and args.skip_existing:
                        mesh = trimesh.load(glb, force="mesh")
                    else:
                        print(f"  [{tag}]")
                        try:
                            mesh = runner.reconstruct(pkl_path, seed, out_glb=glb)
                        except Exception as e:
                            print(f"    FAILED: {type(e).__name__}: {e}")
                            continue
                s = M.score(mesh, tgt, label)
                if s is None:
                    continue
                row = {"name": name, "mode": mode, "level": lv, "condition": cond,
                       "seed": seed, "mesh": str(glb), **s}
                results.append(row)
                results_path.write_text(json.dumps(results, indent=2))
                print(f"    [{tag}] extra={s['extra_pct']:.1f}%  "
                      f"leak={s.get('leak_pct', float('nan')):.1f}%  "
                      f"missing={s['missing_pct']:.1f}%  chamfer={s['chamfer_pct']:.2f}%")

    summarize(results)
    print(f"\nresults: {results_path}")


def summarize(results):
    """Paired clean-vs-clutter summary per (mode, level), with bootstrap CIs."""
    if not results:
        return
    print(f"\n{'mode':7s} {'lvl':>5s} {'n':>3s} | {'Leak% clean':>11s} {'Leak% clut':>10s} "
          f"{'ΔLeak':>7s} {'95% CI':>16s} | {'ΔExtra':>7s} {'95% CI':>16s}")
    print("-" * 104)
    cells = sorted({(r["mode"], r["level"]) for r in results})
    for mode, lv in cells:
        pairs = {}
        for r in results:
            if r["mode"] == mode and r["level"] == lv:
                pairs.setdefault((r["name"], r["seed"]), {})[r["condition"]] = r
        both = [v for v in pairs.values() if "clean" in v and "clutter" in v]
        if not both:
            continue
        ea = [v["clean"]["extra_pct"] for v in both]
        eb = [v["clutter"]["extra_pct"] for v in both]
        # Leak% has a false-positive floor because the intruder rests on the target;
        # the meaningful quantity is the paired RISE over the clean mesh.
        la = [v["clean"].get("leak_pct", np.nan) for v in both]
        lb = [v["clutter"].get("leak_pct", np.nan) for v in both]
        ok = [i for i in range(len(both))
              if not (np.isnan(la[i]) or np.isnan(lb[i]))]
        se = M.paired_bootstrap(ea, eb)
        if ok:
            sl = M.paired_bootstrap([la[i] for i in ok], [lb[i] for i in ok])
            lk = (f"{np.mean([la[i] for i in ok]):11.1f} "
                  f"{np.mean([lb[i] for i in ok]):10.1f} {sl['mean_diff']:+7.1f} "
                  f"[{sl['ci_lo']:+.1f},{sl['ci_hi']:+.1f}]{'*' if sl['significant'] else ' '}")
        else:
            lk = " " * 47
        print(f"{mode:7s} {lv:5.2f} {len(both):3d} | {lk} | {se['mean_diff']:+7.1f} "
              f"[{se['ci_lo']:+.1f},{se['ci_hi']:+.1f}]{'*' if se['significant'] else ' '}")
    print("\n* = paired bootstrap 95% CI excludes zero. "
          "ΔLeak is the rise over the clean mesh scored against the same intruder.")


if __name__ == "__main__":
    main()
