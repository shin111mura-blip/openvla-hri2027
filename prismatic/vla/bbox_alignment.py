"""BBox geometry helpers matching Prismatic image preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch


@dataclass(frozen=True)
class ImageTransformGeometry:
    original_width: int
    original_height: int
    input_width: int
    input_height: int
    resize_strategy: str = "letterbox"


def _as_tensor(bboxes_xyxy: torch.Tensor | Iterable[Iterable[float]]) -> torch.Tensor:
    return bboxes_xyxy if isinstance(bboxes_xyxy, torch.Tensor) else torch.tensor(list(bboxes_xyxy), dtype=torch.float32)


def normalize_xyxy(bboxes_xyxy: torch.Tensor, width: int, height: int) -> torch.Tensor:
    scale = bboxes_xyxy.new_tensor([width, height, width, height]).clamp_min(1.0)
    return (bboxes_xyxy / scale).clamp(0.0, 1.0)


def denormalize_xyxy(bboxes_normalized: torch.Tensor, width: int, height: int) -> torch.Tensor:
    scale = bboxes_normalized.new_tensor([width, height, width, height])
    return bboxes_normalized * scale


def transform_bboxes_to_model_input(
    bboxes_xyxy: torch.Tensor | Iterable[Iterable[float]],
    geometry: ImageTransformGeometry,
) -> torch.Tensor:
    """Apply Prismatic letterbox/resize-crop/resize-naive geometry to BBoxes.

    Output is normalized xyxy in the model input image coordinate system.
    """

    boxes = _as_tensor(bboxes_xyxy).float()
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)

    ow, oh = float(geometry.original_width), float(geometry.original_height)
    iw, ih = float(geometry.input_width), float(geometry.input_height)

    if geometry.resize_strategy == "letterbox":
        side = max(ow, oh)
        pad_x = (side - ow) / 2.0
        pad_y = (side - oh) / 2.0
        boxes = boxes + boxes.new_tensor([pad_x, pad_y, pad_x, pad_y])
        boxes = boxes / side
    elif geometry.resize_strategy == "resize-naive":
        boxes = normalize_xyxy(boxes, int(ow), int(oh))
    elif geometry.resize_strategy == "resize-crop":
        scale = max(iw / ow, ih / oh)
        resized_w, resized_h = ow * scale, oh * scale
        crop_x = max(0.0, (resized_w - iw) / 2.0)
        crop_y = max(0.0, (resized_h - ih) / 2.0)
        boxes = boxes * scale - boxes.new_tensor([crop_x, crop_y, crop_x, crop_y])
        boxes = boxes / boxes.new_tensor([iw, ih, iw, ih])
    else:
        raise ValueError(f"Unsupported resize_strategy={geometry.resize_strategy}")

    return boxes.clamp(0.0, 1.0)


def bbox_xyxy_to_features(bboxes_normalized: torch.Tensor) -> torch.Tensor:
    if bboxes_normalized.numel() == 0:
        return bboxes_normalized.reshape(0, 8)
    x1, y1, x2, y2 = bboxes_normalized.unbind(dim=-1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1).clamp_min(0.0)
    h = (y2 - y1).clamp_min(0.0)
    return torch.stack([x1, y1, x2, y2, cx, cy, w, h], dim=-1)


def bboxes_to_patch_mask(
    bboxes_normalized: torch.Tensor,
    patch_grid: Tuple[int, int],
) -> torch.Tensor:
    """Return mask [num_objects, num_patches] for normalized BBoxes over a regular patch grid."""

    boxes = bboxes_normalized.float()
    gh, gw = patch_grid
    if boxes.numel() == 0:
        return boxes.new_zeros((0, gh * gw), dtype=torch.bool)

    ys = (torch.arange(gh, device=boxes.device, dtype=boxes.dtype) + 0.5) / float(gh)
    xs = (torch.arange(gw, device=boxes.device, dtype=boxes.dtype) + 0.5) / float(gw)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    mask = (
        (centers[:, 0][None, :] >= x1[:, None])
        & (centers[:, 0][None, :] <= x2[:, None])
        & (centers[:, 1][None, :] >= y1[:, None])
        & (centers[:, 1][None, :] <= y2[:, None])
    )

    # Guarantee at least the nearest patch to the BBox center for tiny boxes.
    empty = ~mask.any(dim=1)
    if empty.any():
        cx = ((x1 + x2) / 2.0).clamp(0.0, 1.0 - 1e-6)
        cy = ((y1 + y2) / 2.0).clamp(0.0, 1.0 - 1e-6)
        px = torch.floor(cx * gw).long().clamp(0, gw - 1)
        py = torch.floor(cy * gh).long().clamp(0, gh - 1)
        mask[empty, py[empty] * gw + px[empty]] = True
    return mask


def infer_square_patch_grid(num_patches: int) -> Tuple[int, int]:
    side = int(num_patches**0.5)
    if side * side != num_patches:
        raise ValueError(f"Cannot infer square patch grid from num_patches={num_patches}")
    return side, side
