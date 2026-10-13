# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Stage 2a — is ShapeR's latent set spatially segmentable at all?

Everything downstream depends on this. The proposed method removes clutter by
suppressing latent TOKENS, which only makes sense if a token's influence is
spatially localised. If every token contributed everywhere, no token-level
segmentation could isolate the intruding object and the whole approach is dead.

Why there is reason to expect localisation
------------------------------------------
The VAE is a Michelangelo/VecSet-style perceiver. In `SharpCoarseCrossAttentionEncoder`
with `use_downsample=True`, the latent queries are not free parameters — they are
farthest-point-sampled positions ON the input surface (`fps(pos, batch, ratio=...)`),
embedded and cross-attended against the point cloud. So at ENCODE time token k is
anchored to a specific 3D location by construction.

At GENERATION time the latents come from the flow-matching denoiser instead, out of
noise, with no explicit anchor. Whether the generated set inherits that spatial
organisation is an empirical question, and it is the question this script answers.

Two independent measurements
----------------------------
1. ATTENTION FOOTPRINT. `PerceiverCrossAttentionDecoder` cross-attends 3D query
   points against the latent tokens. Recomputing that attention gives, for each
   query point, a distribution over tokens; transposed, it gives each token a
   spatial footprint over 3D space. A localised token has a tight footprint.
   `QKVMultiheadCrossAttention` calls fused `attention(q,k,v)` and returns no
   weights, so the same projections (`c_q`, `c_kv`) are re-applied here and the
   softmax recomputed explicitly. Note the head split: `c_kv` produces width*2
   channels and is split into k and v with `attn_ch = width // heads // 2`, so the
   query must be sliced to the same channel count.

2. ABLATION DISPLACEMENT. Drop token k, decode, and measure where the occupancy
   field actually changed. This is the ground truth for "what does this token do",
   and it validates measurement 1 rather than trusting attention as explanation.

Outputs a per-token localisation score: the fraction of a token's influence falling
within a sphere of radius `r` around its footprint centroid, versus the same score
for a uniform (non-localised) token.

    python updated_2/latent_probe.py --input_pkl updated_2/outputs/stage1/clutter/ADT1292__woodentable__clean.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from demo_limitation import Runner  # noqa: E402


# --------------------------------------------------------------------------- #
# decoder cross-attention, recomputed
# --------------------------------------------------------------------------- #

def decoder_attention(vae, queries, latents):
    """
    Attention weights of the decoder's cross-attention block.

    Returns (n_queries, n_tokens), averaged over heads.

    Mirrors `QKVMultiheadCrossAttention.forward` exactly. Note the subtlety in that
    function: it rebinds `width` from the KV tensor's own last dim, which is already
    `2 * blk.width` because `c_kv` projects to `width * 2`. So its
    `attn_ch = width // heads // 2` evaluates to `blk.width // heads`, and q — whose
    last dim is `blk.width // heads` after the head split — matches k exactly with
    no truncation. Using `blk.width` directly here instead gives half the true
    channel count and the k/v split then yields four chunks rather than two.
    """
    dec = vae.model.decoder
    blk = dec.cross_attn_decoder.attn      # MultiheadCrossAttention
    heads = blk.heads
    attn_ch = blk.width // heads

    with torch.no_grad():
        q_in = dec.query_proj(dec.embedder(queries))
        # the residual block layer-norms its inputs before attending
        q_in = dec.cross_attn_decoder.ln_1(q_in)
        data = dec.cross_attn_decoder.ln_2(latents)

        q = blk.c_q(q_in)
        kv = blk.c_kv(data)

        b, n, _ = q.shape
        m = kv.shape[1]
        q = q.view(b, n, heads, -1).permute(0, 2, 1, 3)
        kv = kv.view(b, m, heads, -1)
        k, _v = torch.split(kv, attn_ch, dim=-1)
        k = k.permute(0, 2, 1, 3)
        assert q.shape[-1] == k.shape[-1], (
            f"q/k channel mismatch: {q.shape[-1]} vs {k.shape[-1]}")

        logits = torch.einsum("bhnc,bhmc->bhnm", q.float(), k.float())
        logits = logits / np.sqrt(attn_ch)
        attn = logits.softmax(dim=-1)          # (b, heads, n, m)
        return attn.mean(dim=1)[0]             # (n, m)


def grid_queries(bound=1.0, res=48, device="cuda", dtype=torch.bfloat16):
    """Dense query grid in the VAE's normalised cube."""
    g = torch.linspace(-bound, bound, res)
    X, Y, Z = torch.meshgrid(g, g, g, indexing="ij")
    q = torch.stack([X, Y, Z], dim=-1).reshape(1, -1, 3)
    return q.to(device=device, dtype=dtype), (res, res, res)


def surface_queries(vae, latents, res=48, iso=None, max_points=20000):
    """
    Query points ON the predicted surface — where the occupancy/UDF field is near
    its iso level. Attention on empty space is uninformative; the surface is where
    the token actually writes geometry.
    """
    q, shape = grid_queries(res=res, device=latents.device, dtype=latents.dtype)
    with torch.no_grad():
        dec_lat = vae.model.decode(latents)
        vals = []
        for i in range(0, q.shape[1], 65536):
            vals.append(vae.model.query(q[:, i:i + 65536], dec_lat).float())
        vals = torch.cat(vals, dim=1)[0]
    iso = vae.model.udf_iso if iso is None else iso
    band = (vals - iso).abs()
    n = min(max_points, int((band < band.median()).sum()))
    idx = torch.argsort(band)[:n]
    return q[:, idx], vals[idx], dec_lat


def localisation(attn, positions, radius_frac=0.15):
    """
    Per-token localisation score.

    For token k, weights w = attn[:, k] over query positions. Its footprint centroid
    is the weighted mean position; the score is the share of weight inside a sphere
    of `radius_frac` x the query cloud's extent around that centroid. A perfectly
    localised token scores ~1.0; a token spread uniformly over the surface scores
    roughly the volume fraction of that sphere.
    """
    w = attn / attn.sum(dim=0, keepdim=True).clamp_min(1e-12)   # (n, m) per-token
    pos = positions.float()
    ext = float((pos.max(0).values - pos.min(0).values).max())
    r = radius_frac * ext
    cent = (w.T @ pos)                                          # (m, 3)
    d = torch.cdist(cent, pos)                                  # (m, n)
    inside = (d < r).float()
    score = (inside * w.T).sum(dim=1)                           # (m,)
    return score.cpu().numpy(), cent.cpu().numpy(), r, ext


def ablation_check(vae, latents, attn, positions, centroids, n_tokens=8, seed=0):
    """
    Validate the attention footprint against what dropping the token really does.

    For a sample of tokens: zero the token, re-decode at the same query points, and
    find where the field moved most. If the attention footprint is meaningful, the
    displacement centroid should sit near the attention centroid.
    """
    rng = np.random.default_rng(seed)
    m = latents.shape[1]
    pick = rng.choice(m, size=min(n_tokens, m), replace=False)
    pos = positions.float()
    with torch.no_grad():
        base = vae.model.query(positions.unsqueeze(0) if positions.dim() == 2
                               else positions, vae.model.decode(latents)).float()[0]
    rows = []
    for k in pick:
        lat = latents.clone()
        lat[:, k, :] = 0
        with torch.no_grad():
            alt = vae.model.query(positions.unsqueeze(0) if positions.dim() == 2
                                  else positions, vae.model.decode(lat)).float()[0]
        dv = (alt - base).abs()
        if float(dv.sum()) < 1e-9:
            rows.append({"token": int(k), "dead": True})
            continue
        w = dv / dv.sum()
        c = (w[:, None] * pos).sum(dim=0)
        rows.append({
            "token": int(k),
            "dead": False,
            "displacement_centroid": c.cpu().numpy().tolist(),
            "attention_centroid": centroids[k].tolist(),
            "centroid_gap": float(torch.norm(c - torch.tensor(
                centroids[k], device=c.device, dtype=c.dtype))),
            "mean_abs_change": float(dv.mean()),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_pkl", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--radius_frac", type=float, default=0.15)
    ap.add_argument("--n_ablate", type=int, default=8)
    ap.add_argument("--out", default="updated_2/outputs/stage2/latent_probe.json")
    args = ap.parse_args()

    runner = Runner("balance")
    print(f"reconstructing {args.input_pkl} (seed {args.seed}) ...")
    _mesh, latents = runner.reconstruct(args.input_pkl, args.seed, return_latents=True)
    print(f"latent set: {tuple(latents.shape)}")

    q, _vals, dec_lat = surface_queries(runner.vae, latents, res=args.res)
    print(f"surface query points: {q.shape[1]}")

    attn = decoder_attention(runner.vae, q, dec_lat)
    print(f"attention: {tuple(attn.shape)}  (queries x tokens)")

    score, cent, r, ext = localisation(attn, q[0], args.radius_frac)
    # A non-localised baseline: what a uniform token would score is the fraction of
    # query points inside the same sphere, averaged over centroids.
    pos = q[0].float()
    d = torch.cdist(torch.tensor(cent, device=pos.device, dtype=pos.dtype), pos)
    uniform = (d < r).float().mean(dim=1).cpu().numpy()

    print(f"\nlocalisation (radius = {args.radius_frac:.2f} x extent = {r:.3f})")
    print(f"  token score   mean {score.mean():.3f}  median {np.median(score):.3f}  "
          f"p10 {np.percentile(score,10):.3f}  p90 {np.percentile(score,90):.3f}")
    print(f"  uniform ref   mean {uniform.mean():.3f}")
    print(f"  ratio         {score.mean()/max(uniform.mean(),1e-9):.1f}x")
    frac_local = float((score > 3 * uniform).mean())
    print(f"  tokens >3x uniform: {frac_local*100:.0f}%")

    abl = ablation_check(runner.vae, latents, attn, q[0], cent, args.n_ablate)
    live = [r_ for r_ in abl if not r_["dead"]]
    if live:
        gaps = [r_["centroid_gap"] for r_ in live]
        print(f"\nablation check on {len(live)} tokens "
              f"({len(abl)-len(live)} had no effect)")
        print(f"  attention vs displacement centroid gap: "
              f"median {np.median(gaps):.3f}, extent {ext:.3f} "
              f"({np.median(gaps)/ext*100:.0f}% of extent)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "input_pkl": args.input_pkl, "seed": args.seed,
        "n_tokens": int(latents.shape[1]), "n_queries": int(q.shape[1]),
        "radius_frac": args.radius_frac, "radius": r, "extent": ext,
        "localisation_mean": float(score.mean()),
        "localisation_median": float(np.median(score)),
        "uniform_reference_mean": float(uniform.mean()),
        "frac_tokens_above_3x_uniform": frac_local,
        "per_token_score": score.tolist(),
        "per_token_centroid": cent.tolist(),
        "ablation": abl,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
