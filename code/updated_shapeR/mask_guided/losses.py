# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Training objectives for mask-guided ShapeR fine-tuning.

  1. flow_matching_loss   — the rectified-flow objective ShapeR was trained with.
                            Only the mask-guidance params get gradients (backbone frozen).
  2. background_occupancy_loss — the differentiable "silhouette" objective: 3D points that
                            project into SAM2 *background* across the views should be EMPTY.
                            Queried through the VAE decoder (no marching cubes), so it is
                            differentiable. Off by default (weight 0) — see notes.

VERIFY-ON-FIRST-GPU-RUN (documented honestly):
  * target-latent encoding: the VAE's `encode_for_diffusion` expects coarse+sharp surface
    samples with per-point features (normals). Confirm the feature layout and token-count
    argument against how the released checkpoint was trained.
  * occupancy sign convention: `vae.model.query` returns logits; confirm whether >0 means
    inside/occupied (UDF vs occupancy) before trusting the background loss sign.
  * camera convention: we treat batch['camera_extrinsics'] as camera->world (matches
    dataset.image_processor). Confirm before relying on background_occupancy_loss.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import trimesh


# =============================================================================
# Target latent from a GT mesh
# =============================================================================


def sample_surface_with_normals(vertices: np.ndarray, faces: np.ndarray, n: int, device, dtype):
    """Sample `n` surface points + normals from a mesh -> [1, n, 6] (xyz, nx, ny, nz)."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    pts, fid = trimesh.sample.sample_surface(mesh, n)
    nrm = mesh.face_normals[fid]
    surf = np.concatenate([pts, nrm], axis=1).astype(np.float32)
    return torch.from_numpy(surf).to(device=device, dtype=dtype).unsqueeze(0)


@torch.no_grad()
def encode_target_latent(vae, vertices, faces, scale, token_prob_index, n_points, device, dtype):
    """
    Encode a (scaled) GT mesh into the diffusion target latent z1, matching the token
    count used at inference (token_prob_index selects the VAE token scale).
    """
    verts = np.asarray(vertices, dtype=np.float32) * float(scale)
    coarse = sample_surface_with_normals(verts, faces, n_points, device, dtype)
    sharp = sample_surface_with_normals(verts, faces, n_points, device, dtype)
    # force the same token distribution the checkpoint uses at inference
    probs = np.zeros(len(vae.model.get_token_scales()))
    probs[token_prob_index] = 1.0
    z1 = vae.encode_for_diffusion(
        coarse, sharp, is_training=False, force_token_distribution=probs, sample_posterior=False
    )
    return z1  # [1, L, embed_dim]


# =============================================================================
# Flow-matching (rectified flow) loss
# =============================================================================


def flow_matching_loss(model, batch, z1, num_denoise_tokens=None, t_sample="uniform"):
    """
    Rectified-flow loss on target latent z1.
        z0 ~ N(0,1);  t ~ U(0,1);  z_t = (1-t) z0 + t z1;  v* = z1 - z0
        loss = || model(z_t, t, batch) - v* ||^2
    Backbone is frozen; gradients reach only the mask-guidance params.
    """
    B, L, D = z1.shape
    device, dtype = z1.device, z1.dtype
    z0 = torch.randn(B, L, D, device=device, dtype=dtype)
    if t_sample == "uniform":
        t = torch.rand(B, device=device, dtype=torch.float32)
    else:  # logit-normal (flux-like) emphasis on middle timesteps
        t = torch.sigmoid(torch.randn(B, device=device))
    t_ = t.view(B, 1, 1).to(dtype)
    z_t = (1 - t_) * z0 + t_ * z1
    v_target = (z1 - z0).detach()
    v_pred = model(z_t, t.to(dtype), batch)          # denoiser returns velocity
    return F.mse_loss(v_pred.float(), v_target.float())


# =============================================================================
# Differentiable background-occupancy ("silhouette") loss  — optional/experimental
# =============================================================================


def _project_points(points_w, extrinsics_c2w, intrinsics, H, W):
    """Project [P,3] world points into a pinhole view -> [P,2] uv + validity mask."""
    w2c = torch.linalg.inv(extrinsics_c2w)
    ones = torch.ones(points_w.shape[0], 1, device=points_w.device, dtype=points_w.dtype)
    ph = torch.cat([points_w, ones], dim=1) @ w2c.T
    cam = ph[:, :3]
    z = cam[:, 2:3].clamp_min(1e-4)
    uv = (cam @ intrinsics[:3, :3].T)
    uv = uv[:, :2] / z
    valid = (cam[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv, valid


def background_occupancy_loss(model, vae, batch, z_t, v_pred, t, fg_masks, n_query=4096, occupied_is_positive=True):
    """
    Points that project into SAM2 background across the views should be empty.

    One-step estimate z1_hat = z_t + (1-t) v_pred -> decode -> query occupancy at
    background 3D points -> penalise occupancy. Differentiable through the decoder.

    Returns a scalar loss (0 if no background points were found). EXPERIMENTAL: verify the
    occupancy sign + camera convention (see module notes) before weighting this up.
    """
    device = z_t.device
    B = z_t.shape[0]
    if B != 1:
        raise NotImplementedError("background_occupancy_loss currently supports batch size 1.")

    t_ = float(t.view(-1)[0]) if torch.is_tensor(t) else float(t)
    z1_hat = z_t + (1.0 - t_) * v_pred

    # candidate 3D points in the normalized cube
    pts = (torch.rand(n_query, 3, device=device) * 2 - 1) * 0.95

    ext = batch["camera_extrinsics"].reshape(-1, 4, 4).float()   # [V,4,4] cam->world
    intr = batch["camera_intrinsics"].reshape(-1, 4, 4).float()  # [V,4,4]
    V = ext.shape[0]
    H = W = fg_masks.shape[-1]
    fg = fg_masks.reshape(V, 1, H, W).float()

    in_frame = torch.zeros(n_query, device=device)
    in_bg = torch.zeros(n_query, device=device)
    for v in range(V):
        uv, valid = _project_points(pts.float(), ext[v], intr[v], H, W)
        if valid.sum() == 0:
            continue
        xs = uv[:, 0].round().long().clamp(0, W - 1)
        ys = uv[:, 1].round().long().clamp(0, H - 1)
        fgv = fg[v, 0][ys, xs]                    # 1 = foreground, 0 = background
        in_frame += valid.float()
        in_bg += (valid & (fgv < 0.5)).float()

    # background point = in frame somewhere and (almost) never inside the foreground
    bg_mask = (in_frame > 0) & (in_bg >= (in_frame - 1e-3))
    if bg_mask.sum() == 0:
        return z_t.new_zeros(())

    bg_pts = pts[bg_mask].unsqueeze(0)           # [1, M, 3]
    # training runs the VAE in fp32 (encoder fps needs it), so decode/query in fp32 too
    vae_dtype = next(vae.model.parameters()).dtype
    latents = vae.model.decode(z1_hat.to(vae_dtype))
    occ = vae.model.query(bg_pts.to(vae_dtype), latents).float()   # [1, M] logits
    if not occupied_is_positive:
        occ = -occ
    # penalise occupancy at background points (push logits negative / empty)
    return F.softplus(occ).mean()
