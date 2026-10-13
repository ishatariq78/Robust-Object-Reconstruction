# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Mask-guided ShapeR — Option A + Option B, built by SUBCLASSING the frozen ShapeR
model. Nothing in the original ShapeR package is modified.

  Option A  (foreground mask channel):
      ShapeR's DINO branch already has a trained 1-channel mask embedder
      (`_mask_patch_embed`, fed the point-projection mask). We widen its first conv
      1 -> 2 channels: channel 0 = point-projection mask (trained weights kept),
      channel 1 = SAM2 foreground mask (weights ZERO-INITIALISED). At init the model
      is byte-for-byte the pretrained one; fine-tuning teaches it to use channel 1.

  Option B  (foreground attention bias, Amodal3R-style):
      For every image condition token we compute a per-patch foreground weight
      w in [0,1] (average-pool the SAM2 mask onto the DINO patch grid). Inside the
      denoiser's attention we add a bias  lambda * log(w + eps)  to the attention
      logits of the image-token KEYS, so background patches are attended less. `lambda`
      (`fg_bias_scale`) is a single learnable scalar, ZERO-INITIALISED, so again the
      model starts identical to pretrained and learns how hard to push.

Both use zero-init, so a freshly-built mask-guided model reproduces base ShapeR exactly
(see selftest.py). Fine-tuning trains only the new parameters (see train.py).
"""

from __future__ import annotations

import copy
import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.flow_matching.shaper_denoiser import ShapeRDenoiser
from torchsparse import SparseTensor


# =============================================================================
# Bias-aware attention (SDPA supports an additive attn_mask == bias on the logits)
# =============================================================================


def attention_with_bias(q, k, v, bias=None):
    """q,k,v: [B, H, L, D]. bias: additive logit bias broadcastable to [B,H,Lq,Lk] or None."""
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    return rearrange(x, "B H L D -> B L (H D)")


def _double_block_forward_biased(self, img, txt, vec):
    """Copy of DoubleStreamBlock.forward, but the joint attention adds self._mg_bias."""
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)

    img_modulated = self.img_norm1(img)
    img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
    img_qkv = self.img_attn.qkv(img_modulated)
    img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

    txt_modulated = self.txt_norm1(txt)
    txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

    # joint sequence order: [txt (condition ctx) ; img (latent)]
    q = torch.cat((txt_q, img_q), dim=2)
    k = torch.cat((txt_k, img_k), dim=2)
    v = torch.cat((txt_v, img_v), dim=2)

    bias = getattr(self, "_mg_bias", None)
    attn = attention_with_bias(q, k, v, bias)
    txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

    img = img + img_mod1.gate * self.img_attn.proj(img_attn)
    img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
    txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
    txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
    return img, txt


def _single_block_forward_biased(self, x, vec):
    """Copy of SingleStreamBlock.forward, but the attention adds self._mg_bias."""
    mod, _ = self.modulation(vec)
    x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
    qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
    q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    q, k = self.norm(q, k, v)
    bias = getattr(self, "_mg_bias", None)
    attn = attention_with_bias(q, k, v, bias)
    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
    return x + mod.gate * output


def patch_transformer_blocks(transformer) -> None:
    """Rebind each block's forward to the bias-aware version (reads block._mg_bias)."""
    for blk in transformer.double_blocks:
        blk._mg_bias = None
        blk.forward = types.MethodType(_double_block_forward_biased, blk)
    for blk in transformer.single_blocks:
        blk._mg_bias = None
        blk.forward = types.MethodType(_single_block_forward_biased, blk)


# =============================================================================
# Option A: widen the mask embedder's first conv from 1 -> 2 channels (zero-init ch1)
# =============================================================================


def widen_mask_embedder(dino_extractor) -> None:
    """
    Replace `_mask_patch_embed[0].proj_in` (a Conv2d(1, C, 3)) with a Conv2d(2, C, 3):
    channel 0 keeps the trained weights, channel 1 is zero-init.
    Idempotent (skips if already 2-channel).
    """
    mask_net = dino_extractor._mask_patch_embed[0]  # MaskDownsamplingNet
    first = mask_net.proj_in                        # nn.Conv2d(1, base_channels, 3, padding=1)
    if first.in_channels == 2:
        return
    new = nn.Conv2d(2, first.out_channels, kernel_size=first.kernel_size,
                    stride=first.stride, padding=first.padding, bias=first.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :1].copy_(first.weight)         # channel 0 = trained point-mask
        if first.bias is not None:
            new.bias.copy_(first.bias)
    new = new.to(first.weight.device, first.weight.dtype)
    mask_net.proj_in = new


# =============================================================================
# Mask-guided denoiser
# =============================================================================


class MaskGuidedDenoiser(ShapeRDenoiser):
    """
    ShapeRDenoiser + Option A (2-channel mask) + Option B (foreground attention bias).

    Set `self._fg_masks` to [B*V, 1, H, W] foreground masks (aligned to the reshaped
    views, same order ShapeR uses) before each forward. `fg_bias_scale` is the learnable
    lambda for Option B.
    """

    def __init__(self, config):
        super().__init__(config)
        # Option B: single learnable bias scale, zero-init -> no effect at start.
        self.fg_bias_scale = nn.Parameter(torch.zeros(1))
        self._fg_masks = None
        self._use_fg_channel = True   # Option A on
        self._use_fg_bias = True      # Option B on

    # ---- per-image-token foreground weights --------------------------------
    @staticmethod
    def _image_token_weights(fg_bv: torch.Tensor, num_views: int, n_image_tokens: int) -> torch.Tensor:
        """fg_bv [B*V,1,H,W] -> w [B, n_image_tokens] in img_tokens order ([patches..., cls] per view)."""
        bv = fg_bv.shape[0]
        b = bv // num_views
        tpv = n_image_tokens // num_views
        # detect grid size and whether a cls token is appended
        if int(math.isqrt(tpv)) ** 2 == tpv:
            g, has_cls = int(math.isqrt(tpv)), False
        elif int(math.isqrt(tpv - 1)) ** 2 == (tpv - 1):
            g, has_cls = int(math.isqrt(tpv - 1)), True
        else:
            g, has_cls = int(round(math.sqrt(tpv))), (int(round(math.sqrt(tpv))) ** 2 != tpv)
        pooled = F.adaptive_avg_pool2d(fg_bv.float(), (g, g)).reshape(bv, g * g)
        if has_cls:
            pooled = torch.cat([pooled, pooled.mean(dim=1, keepdim=True)], dim=1)  # cls = mean fg
        return pooled.reshape(b, num_views * tpv)

    def _set_attention_bias(self, w_img, cond_len, img_start, latent_len, dtype, device):
        """Build [B,1,1,cond_len+latent_len] additive bias and install on every block."""
        b, n_img = w_img.shape
        total = cond_len + latent_len
        logw = torch.log(w_img.clamp_min(0.0) + 1e-4)          # [B, n_img]
        bias_full = torch.zeros(b, total, dtype=torch.float32, device=device)
        bias_full[:, img_start : img_start + n_img] = logw
        bias = (self.fg_bias_scale.float() * bias_full).to(dtype).view(b, 1, 1, total)
        for blk in self.transformer.double_blocks:
            blk._mg_bias = bias
        for blk in self.transformer.single_blocks:
            blk._mg_bias = bias

    def _clear_attention_bias(self):
        for blk in list(self.transformer.double_blocks) + list(self.transformer.single_blocks):
            blk._mg_bias = None

    # ---- forward (copy of ShapeRDenoiser.forward_impl with A + B injected) ---
    def forward_impl(self, z_x_t, pc_cond, img_in, msk_in, box_in, cam_ext, cam_int,
                     txt_in_t5, txt_in_clip, t, skipped_condition, return_intermediate=False):
        z_x_t_in = self.simple_latent_projection(z_x_t)
        B = z_x_t_in.shape[0]

        condition_context = []
        clip_tokens = None
        img_start = None
        n_image_tokens = 0
        cond_len_before_image = 0
        running_len = 0

        if "point" in self.input_types:
            if skipped_condition is not None and "point" in skipped_condition:
                point_tokens = self.null_context.expand(B, -1, -1)
                condition_context.append(point_tokens)
                condition_context.append(self.point_cond_separator.expand(B, -1, -1))
            else:
                pc_features = self.point_cloud_feature_extractor(pc_cond)
                pc_ctx = pc_features["context"].reshape(B, -1, pc_features["context"].shape[-1]).to(z_x_t_in.dtype)
                condition_context.append(pc_ctx)
                condition_context.append(self.point_cond_separator.expand(pc_ctx.shape[0], -1, -1))
        running_len = sum(c.shape[1] for c in condition_context)

        if "image" in self.input_types:
            if skipped_condition is not None and "image" in skipped_condition:
                img_tokens = self.null_context.expand(B, -1, -1)
                condition_context.append(img_tokens)
                condition_context.append(self.image_cond_separator.expand(B, -1, -1))
            else:
                Bv, V, C, H, W = img_in.shape
                img_in = img_in.reshape(B * V, C, H, W)
                mask1 = msk_in.reshape(B * V, 1, H, W)
                box_in = box_in.reshape(B * V, 2, 2)
                camera_extrinsics = cam_ext.reshape(B * V, 4, 4)
                camera_intrinsics = cam_int.reshape(B * V, 4, 4)

                # ---- Option A: feed [point-projection mask ; SAM2 fg mask] ----
                masks_for_dino = mask1
                if self._use_fg_channel and self._fg_masks is not None:
                    fg = self._fg_masks.to(mask1.dtype).reshape(B * V, 1, H, W)
                    masks_for_dino = torch.cat([mask1, fg], dim=1)  # [B*V, 2, H, W]

                dino_feats = self.dino_ray_extractor(
                    img_in, camera_extrinsics, camera_intrinsics, masks_for_dino, box_in
                )
                img_tokens = self.dino_ray_extractor.get_token_feature(dino_feats)  # [B*V, Tpv, feat]
                img_tokens = img_tokens.reshape(B, -1, img_tokens.shape[-1])        # [B, V*Tpv, feat]

                img_start = running_len
                n_image_tokens = img_tokens.shape[1]
                condition_context.append(img_tokens)
                condition_context.append(self.image_cond_separator.expand(img_tokens.shape[0], -1, -1))

        txt_tokens = None
        if "text" in self.input_types:
            if skipped_condition is not None and "text" in skipped_condition:
                txt_tokens = self.null_context.expand(B, -1, -1)
                condition_context.append(txt_tokens)
                condition_context.append(self.text_cond_separator.expand(B, -1, -1))
            else:
                txt_tokens = self.simple_t5_projection(txt_in_t5)
                if not self.use_pre_text_attn_blocks:
                    condition_context.append(txt_tokens)
                else:
                    condition_context.append(self.null_context.expand(B, -1, -1))
                condition_context.append(self.text_cond_separator.expand(condition_context[-1].shape[0], -1, -1))
                clip_tokens = self.simple_clip_projection(txt_in_clip)

        condition_context = torch.cat(condition_context, dim=1) if condition_context else None

        # ---- Option B: install the foreground attention bias ----
        bias_installed = False
        if (self._use_fg_bias and self._fg_masks is not None and img_start is not None
                and condition_context is not None):
            V = img_in.shape[0] // B if img_in is not None else 1
            w_img = self._image_token_weights(self._fg_masks.to(z_x_t_in.device), V, n_image_tokens)
            self._set_attention_bias(
                w_img, cond_len=condition_context.shape[1], img_start=img_start,
                latent_len=z_x_t_in.shape[1], dtype=z_x_t_in.dtype, device=z_x_t_in.device,
            )
            bias_installed = True

        try:
            output = self.transformer(z_x_t_in, condition_context, t, y=clip_tokens, txt_tokens=txt_tokens)
        finally:
            if bias_installed:
                self._clear_attention_bias()

        output, intermediate = output
        if return_intermediate:
            return output, intermediate
        return output


# =============================================================================
# Builder: load pretrained ShapeR into the mask-guided model, apply A + B
# =============================================================================


def build_mask_guided_denoiser(config, ckpt_path, device, use_fg_channel=True, use_fg_bias=True):
    """Construct MaskGuidedDenoiser, load pretrained ShapeR weights, apply Option A/B (zero-init)."""
    model = MaskGuidedDenoiser(config).to(device)
    model.convert_to_bfloat16()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # `fg_bias_scale` (+ widened conv, added below) are the only expected 'missing' keys.
    model._use_fg_channel = use_fg_channel
    model._use_fg_bias = use_fg_bias
    if use_fg_channel and "image" in model.input_types:
        widen_mask_embedder(model.dino_ray_extractor)   # zero-init 2nd channel
    if use_fg_bias:
        patch_transformer_blocks(model.transformer)     # bias-aware attention
    model.fg_bias_scale.data.zero_()                     # ensure lambda == 0 at start
    return model, missing, unexpected


def trainable_parameters(model, train_bias=True, train_mask_channel=True):
    """Freeze the whole backbone; return only the new mask-guidance parameters to optimise."""
    for p in model.parameters():
        p.requires_grad = False
    params = []
    if train_bias:
        model.fg_bias_scale.requires_grad = True
        params.append(model.fg_bias_scale)
    if train_mask_channel and "image" in model.input_types:
        conv = model.dino_ray_extractor._mask_patch_embed[0].proj_in
        conv.weight.requires_grad = True
        params.append(conv.weight)
        if conv.bias is not None:
            conv.bias.requires_grad = True
            params.append(conv.bias)
    return params
