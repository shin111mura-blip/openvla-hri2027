"""ROI pooling over visual patch features."""

from __future__ import annotations

import torch

from prismatic.vla.bbox_alignment import bboxes_to_patch_mask, infer_square_patch_grid


def roi_mean_pool(
    patch_features: torch.Tensor,
    bboxes_normalized: torch.Tensor,
    object_mask: torch.Tensor,
    patch_grid: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Mean-pool patch features inside each BBox.

    Args:
        patch_features: [batch, num_patches, hidden_dim]
        bboxes_normalized: [batch, max_objects, 4]
        object_mask: [batch, max_objects]
    """

    batch, num_patches, hidden_dim = patch_features.shape
    _, max_objects, _ = bboxes_normalized.shape
    patch_grid = infer_square_patch_grid(num_patches) if patch_grid is None else patch_grid
    pooled = patch_features.new_zeros((batch, max_objects, hidden_dim))
    for b in range(batch):
        mask = bboxes_to_patch_mask(bboxes_normalized[b], patch_grid).to(patch_features.device)
        weights = mask.float()
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled[b] = weights @ patch_features[b] / denom
    return pooled * object_mask.unsqueeze(-1).to(pooled.dtype)
