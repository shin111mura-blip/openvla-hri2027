"""Token layout helpers for optional Object/BBox tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

IGNORE_INDEX = -100


@dataclass(frozen=True)
class TokenLayout:
    num_visual_tokens: int
    num_object_tokens: int
    language_start: int
    action_start: Optional[int]
    object_start: int


def build_multimodal_with_optional_object_tokens(
    input_embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    projected_patch_embeddings: torch.Tensor,
    object_token_embeddings: Optional[torch.Tensor] = None,
    object_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], TokenLayout]:
    batch = input_embeddings.shape[0]
    num_visual = projected_patch_embeddings.shape[1]
    if object_token_embeddings is None:
        object_token_embeddings = input_embeddings.new_zeros((batch, 0, input_embeddings.shape[-1]))
    num_object = object_token_embeddings.shape[1]

    embeddings = torch.cat(
        [input_embeddings[:, :1, :], projected_patch_embeddings, object_token_embeddings, input_embeddings[:, 1:, :]],
        dim=1,
    )

    out_attention = None
    if attention_mask is not None:
        visual_attention = torch.ones((batch, num_visual), dtype=attention_mask.dtype, device=attention_mask.device)
        if num_object:
            if object_mask is None:
                object_attention = torch.ones((batch, num_object), dtype=attention_mask.dtype, device=attention_mask.device)
            else:
                object_attention = object_mask.to(dtype=attention_mask.dtype, device=attention_mask.device)
        else:
            object_attention = attention_mask.new_zeros((batch, 0))
        out_attention = torch.cat([attention_mask[:, :1], visual_attention, object_attention, attention_mask[:, 1:]], dim=1)

    out_labels = None
    action_start = None
    if labels is not None:
        visual_labels = torch.full((batch, num_visual), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
        object_labels = torch.full((batch, num_object), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
        out_labels = torch.cat([labels[:, :1], visual_labels, object_labels, labels[:, 1:]], dim=1)
        action_positions = torch.nonzero(out_labels[0] != IGNORE_INDEX, as_tuple=False)
        if action_positions.numel() > 0:
            action_start = int(action_positions[0].item())

    layout = TokenLayout(
        num_visual_tokens=num_visual,
        num_object_tokens=num_object,
        language_start=1 + num_visual + num_object,
        action_start=action_start,
        object_start=1 + num_visual,
    )
    return embeddings, out_attention, out_labels, layout
