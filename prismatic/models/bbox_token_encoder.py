"""Object/BBox token encoder for OpenVLA fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from prismatic.vla.bbox_alignment import bbox_xyxy_to_features
from prismatic.vla.roi_pooling import roi_mean_pool


@dataclass
class BBoxTokenConfig:
    enabled: bool = False
    max_objects: int = 12
    include_coordinates: bool = True
    include_category_embedding: bool = False
    include_confidence_embedding: bool = False
    pooling: str = "mean"
    sort_order: str = "spatial"
    num_categories: int = 0


class BBoxTokenEncoder(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, config: BBoxTokenConfig | None = None) -> None:
        super().__init__()
        self.config = config or BBoxTokenConfig()
        self.roi_projection = nn.Linear(vision_dim, llm_dim)
        self.coord_projection = nn.Sequential(
            nn.Linear(8, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        self.object_type_embedding = nn.Embedding(1, llm_dim)
        self.category_embedding = (
            nn.Embedding(max(1, self.config.num_categories), llm_dim)
            if self.config.include_category_embedding
            else None
        )
        self.confidence_projection = nn.Linear(1, llm_dim) if self.config.include_confidence_embedding else None
        self.layer_norm = nn.LayerNorm(llm_dim)

    def forward(
        self,
        patch_features: torch.Tensor,
        bboxes_normalized: torch.Tensor,
        object_mask: torch.Tensor,
        category_ids: Optional[torch.Tensor] = None,
        confidences: Optional[torch.Tensor] = None,
        patch_grid: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if self.config.pooling != "mean":
            raise ValueError(f"Unsupported pooling={self.config.pooling}")

        roi_features = roi_mean_pool(patch_features, bboxes_normalized, object_mask, patch_grid=patch_grid)
        token = self.roi_projection(roi_features)

        if self.config.include_coordinates:
            coord_features = bbox_xyxy_to_features(bboxes_normalized)
            token = token + self.coord_projection(coord_features.to(token.dtype))

        type_ids = torch.zeros((*object_mask.shape,), dtype=torch.long, device=object_mask.device)
        token = token + self.object_type_embedding(type_ids)

        if self.category_embedding is not None:
            if category_ids is None:
                category_ids = torch.zeros_like(type_ids)
            token = token + self.category_embedding(category_ids.clamp_min(0))

        if self.confidence_projection is not None:
            if confidences is None:
                confidences = torch.zeros((*object_mask.shape, 1), dtype=token.dtype, device=token.device)
            elif confidences.ndim == 2:
                confidences = confidences.unsqueeze(-1)
            token = token + self.confidence_projection(confidences.to(token.dtype))

        return self.layer_norm(token) * object_mask.unsqueeze(-1).to(token.dtype)


def pad_bbox_inputs(
    bboxes: torch.Tensor,
    max_objects: int,
    category_ids: Optional[torch.Tensor] = None,
    confidences: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Pad or truncate BBox tensors with a fixed spatial-order rule."""

    if bboxes.ndim == 2:
        bboxes = bboxes.unsqueeze(0)
    batch, num_objects, _ = bboxes.shape
    keep = min(num_objects, max_objects)
    out_boxes = bboxes.new_zeros((batch, max_objects, 4))
    out_mask = torch.zeros((batch, max_objects), dtype=torch.bool, device=bboxes.device)
    out_boxes[:, :keep] = bboxes[:, :keep]
    out_mask[:, :keep] = True
    out = {"bboxes_normalized": out_boxes, "object_mask": out_mask}
    if category_ids is not None:
        if category_ids.ndim == 1:
            category_ids = category_ids.unsqueeze(0)
        padded = torch.zeros((batch, max_objects), dtype=torch.long, device=bboxes.device)
        padded[:, :keep] = category_ids[:, :keep]
        out["category_ids"] = padded
    if confidences is not None:
        if confidences.ndim == 1:
            confidences = confidences.unsqueeze(0)
        padded_conf = bboxes.new_zeros((batch, max_objects))
        padded_conf[:, :keep] = confidences[:, :keep]
        out["confidences"] = padded_conf
    return out
