# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Foreground-mask conditioning for ShapeR inference (background-masking / mechanism "A").

Goal: stop ShapeR from reconstructing extra background objects. ShapeR conditions its
shape generation partly on DINO features of each input view, and those views are *full
crops that still contain background clutter* (a neighbouring chair, the table, the wall).
DINO therefore encodes that clutter and it can leak into the generated mesh.

This module removes that leak at inference time on the FROZEN model:

  1. For each of ShapeR's cropped input views, run SAM2 to get a foreground (object)
     mask. SAM2 is prompted with the object's SLAM-point projection mask + its box,
     which ShapeR already computes per crop (`masks_ingest` / `boxes_ingest`) and which
     is therefore already aligned to the exact 280x280 crop DINO sees. No re-alignment.

  2. Monkeypatch the DINO feature extractor so that, just before an image is fed to
     DINO, every background pixel (foreground mask == 0) is replaced with a neutral gray.
     DINO then simply never sees the background objects, so they cannot condition the
     shape. This is the "object-only image, background replaced with neutral pixels"
     branch of the design and needs no retraining.

Nothing else in the ShapeR pipeline changes: point-cloud, text, camera-ray, and the
existing point-projection mask conditioning all stay exactly as trained.
"""

from __future__ import annotations

import types

import numpy as np
import torch
from PIL import Image


# =============================================================================
# 1. SAM2 foreground masks on ShapeR's cropped views
# =============================================================================


def _view_tensor_to_pil(img_chw: torch.Tensor) -> Image.Image:
    """[3, H, W] float image in [0, 1] (any dtype/device) -> RGB PIL uint8 image."""
    arr = (img_chw.detach().float().clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _positive_points_from_point_mask(point_mask_hw: torch.Tensor, max_points: int) -> np.ndarray:
    """Sample up to `max_points` (x, y) positive clicks from the SLAM point-projection mask."""
    ys, xs = torch.where(point_mask_hw > 0.5)
    if ys.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = torch.stack([xs, ys], dim=1).float().cpu().numpy()
    if pts.shape[0] > max_points:
        stride = pts.shape[0] / max_points
        pts = pts[(np.arange(max_points) * stride).astype(int)]
    return pts.astype(np.float32)


def compute_view_foreground_masks(
    images_bv: torch.Tensor,
    point_masks_bv: torch.Tensor,
    boxes_bv: torch.Tensor,
    sam2,
    max_points: int = 16,
) -> torch.Tensor:
    """
    Run SAM2 on every cropped view to produce a per-view foreground mask.

    Args:
        images_bv:      [N, 3, H, W] cropped view images in [0, 1] (N = batch*views).
        point_masks_bv: [N, 1, H, W] SLAM point-projection masks (ShapeR's `masks_ingest`).
        boxes_bv:       [N, 2, 2]   object boxes as [[x_min, y_min], [x_max, y_max]]
                                     (ShapeR's `boxes_ingest`).
        sam2:           a SAM2ForegroundSegmenter instance.
        max_points:     max positive click prompts sampled from the point mask.

    Returns:
        [N, 1, H, W] float foreground masks in {0.0, 1.0}, on the same device/dtype as images.
    """
    n, _, h, w = images_bv.shape
    fg = torch.zeros((n, 1, h, w), dtype=images_bv.dtype, device=images_bv.device)

    for i in range(n):
        image = _view_tensor_to_pil(images_bv[i])
        pos_pts = _positive_points_from_point_mask(point_masks_bv[i, 0], max_points)

        box = boxes_bv[i].reshape(-1).float().cpu().numpy()  # [x_min, y_min, x_max, y_max]
        # guard against a degenerate box (no points -> full-frame fallback in ShapeR)
        if not (box[2] > box[0] and box[3] > box[1]):
            box = np.array([0, 0, w - 1, h - 1], dtype=np.float32)

        if pos_pts.shape[0] == 0:
            # no SLAM points in this crop: fall back to a center click inside the box
            pos_pts = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]], dtype=np.float32)

        mask, _score = sam2.segment(
            image,
            positive_points_xy=pos_pts,
            negative_points_xy=None,
            box_xyxy=box,
            clip_to_box=True,
            speckle_removal=False,
        )
        fg[i, 0] = torch.from_numpy(mask.astype(np.float32)).to(fg.device, fg.dtype)

    return fg


# =============================================================================
# 2. Mask-conditioning monkeypatch on the DINO feature extractor
# =============================================================================
#
# Three ways to feed the SAM2 foreground mask to the FROZEN model. All wrap the DINO
# extractor's forward; pick with `mode`:
#
#   "background"   — replace background PIXELS of the image with neutral gray, leave the
#                    mask channel as-is (the original background-masking experiment).
#   "replace_mask" — leave the ORIGINAL image untouched; put the SAM2 foreground mask
#                    into ShapeR's existing mask channel, replacing the point-projection
#                    mask. i.e. "give the model the original image AND the mask".
#   "add_mask"     — leave the ORIGINAL image untouched; add the SAM2 foreground mask on
#                    TOP of the point-projection mask in the mask channel (union), so the
#                    model keeps its trained point cue and also gets the filled foreground.
#
# ShapeR's DINO extractor already has a trained 1-channel mask embedder (`_mask_patch_embed`,
# active because config.encoder.dino_mask_inject=True). The mask-channel modes reuse it, so
# they are less out-of-distribution than graying the RGB image.


def install_mask_conditioning(
    dino_extractor,
    foreground_masks: torch.Tensor,
    mode: str = "background",
    neutral_value: float = 0.5,
) -> None:
    """
    Wrap `dino_extractor.forward` to inject the SAM2 foreground mask according to `mode`
    (see module notes above). `foreground_masks` is [N, 1, H, W] aligned to the
    N = batch*views images/masks the extractor receives (same order ShapeR reshapes them).

    Idempotent: re-installing updates the masks/mode rather than stacking wrappers.
    """
    if mode not in ("background", "replace_mask", "add_mask"):
        raise ValueError(f"Unknown mask mode: {mode!r}")
    dino_extractor._fg_masks = foreground_masks
    dino_extractor._fg_mode = mode
    dino_extractor._fg_neutral_value = float(neutral_value)

    if getattr(dino_extractor, "_fg_orig_forward", None) is None:
        dino_extractor._fg_orig_forward = dino_extractor.forward

        def _patched_forward(self, imgs, camera_to_world, camera_intrinsics, masks=None, boxes=None):
            fg = getattr(self, "_fg_masks", None)
            if fg is not None:
                if fg.shape[0] != imgs.shape[0] or fg.shape[-2:] != imgs.shape[-2:]:
                    raise ValueError(
                        f"Foreground mask shape {tuple(fg.shape)} does not match image batch "
                        f"{tuple(imgs.shape)}; expected [N,1,H,W] aligned to the views."
                    )
                fg = fg.to(dtype=imgs.dtype, device=imgs.device)
                m = self._fg_mode
                if m == "background":
                    # original image gets its background blanked; mask channel unchanged
                    imgs = imgs * fg + self._fg_neutral_value * (1.0 - fg)
                elif m == "replace_mask":
                    # original image untouched; SAM2 mask REPLACES the mask channel
                    masks = fg
                elif m == "add_mask":
                    # original image untouched; SAM2 mask added to the mask channel (union)
                    if masks is not None:
                        masks = torch.clamp(masks.to(fg.dtype) + fg, 0.0, 1.0)
                    else:
                        masks = fg
            return self._fg_orig_forward(imgs, camera_to_world, camera_intrinsics, masks, boxes)

        dino_extractor.forward = types.MethodType(_patched_forward, dino_extractor)


# Backward-compatible alias for the original background-masking entry point.
def install_background_masking(dino_extractor, foreground_masks: torch.Tensor, neutral_value: float = 0.5) -> None:
    install_mask_conditioning(dino_extractor, foreground_masks, mode="background", neutral_value=neutral_value)


def remove_mask_conditioning(dino_extractor) -> None:
    """Undo install_mask_conditioning, restoring the original forward."""
    if getattr(dino_extractor, "_fg_orig_forward", None) is not None:
        dino_extractor.forward = dino_extractor._fg_orig_forward
        dino_extractor._fg_orig_forward = None
    dino_extractor._fg_masks = None


remove_background_masking = remove_mask_conditioning  # backward-compatible alias


# =============================================================================
# 3. Debug: render the background-masked views ShapeR will actually see
# =============================================================================


def make_masked_view_grid(images_bv: torch.Tensor, foreground_masks: torch.Tensor, neutral_value: float = 0.5) -> Image.Image:
    """Return a single horizontal PIL grid of the background-masked views, for inspection."""
    fg = foreground_masks.to(images_bv.dtype)
    masked = images_bv * fg + neutral_value * (1.0 - fg)
    tiles = [ _view_tensor_to_pil(masked[i]) for i in range(masked.shape[0]) ]
    h = tiles[0].height
    grid = Image.new("RGB", (sum(t.width for t in tiles), h))
    x = 0
    for t in tiles:
        grid.paste(t, (x, 0))
        x += t.width
    return grid
