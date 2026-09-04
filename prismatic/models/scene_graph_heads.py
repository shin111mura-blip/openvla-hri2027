"""Auxiliary scene graph prediction heads."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from prismatic.vla.scene_graph.schema import PAIRWISE_RELATIONS


@dataclass(frozen=True)
class SceneGraphHeadOutput:
    edge_logits: torch.Tensor
    between_logits: torch.Tensor
    pair_indices: torch.Tensor
    triplet_indices: torch.Tensor


class PairwiseEdgeHead(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int = len(PAIRWISE_RELATIONS), mlp_dim: Optional[int] = None) -> None:
        super().__init__()
        mlp_dim = mlp_dim or hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_relations),
        )

    def forward(self, object_hidden: torch.Tensor, pair_indices: torch.Tensor) -> torch.Tensor:
        hi = object_hidden[:, pair_indices[:, 0]]
        hj = object_hidden[:, pair_indices[:, 1]]
        return self.net(torch.cat([hi, hj, hi - hj, hi * hj], dim=-1))


class BetweenHead(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: Optional[int] = None) -> None:
        super().__init__()
        mlp_dim = mlp_dim or hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 1),
        )

    def forward(self, object_hidden: torch.Tensor, triplet_indices: torch.Tensor) -> torch.Tensor:
        ht = object_hidden[:, triplet_indices[:, 0]]
        hj = object_hidden[:, triplet_indices[:, 1]]
        hk = object_hidden[:, triplet_indices[:, 2]]
        symmetric = torch.cat([ht, hj + hk, torch.abs(hj - hk), hj * hk], dim=-1)
        return self.net(symmetric).squeeze(-1)


class SceneGraphAuxiliaryHeads(nn.Module):
    """Pairwise and between heads over object-level hidden states only.

    Callers must pass object/ROI hidden states. The module intentionally has no
    action-hidden-state argument to prevent action leakage.
    """

    def __init__(self, hidden_dim: int, mlp_dim: Optional[int] = None) -> None:
        super().__init__()
        self.edge_head = PairwiseEdgeHead(hidden_dim, mlp_dim=mlp_dim)
        self.between_head = BetweenHead(hidden_dim, mlp_dim=mlp_dim)

    @staticmethod
    def make_pair_indices(max_objects: int, device: torch.device) -> torch.Tensor:
        pairs = [(i, j) for i in range(max_objects) for j in range(max_objects) if i != j]
        return torch.tensor(pairs, dtype=torch.long, device=device)

    @staticmethod
    def make_triplet_indices(max_objects: int, device: torch.device) -> torch.Tensor:
        triplets = []
        for target in range(max_objects):
            for ref1, ref2 in combinations([idx for idx in range(max_objects) if idx != target], 2):
                triplets.append((target, ref1, ref2))
        return torch.tensor(triplets, dtype=torch.long, device=device)

    def forward(self, object_hidden: torch.Tensor, object_mask: torch.Tensor) -> SceneGraphHeadOutput:
        max_objects = object_hidden.shape[1]
        pair_indices = self.make_pair_indices(max_objects, object_hidden.device)
        triplet_indices = self.make_triplet_indices(max_objects, object_hidden.device)
        edge_logits = self.edge_head(object_hidden, pair_indices)
        between_logits = self.between_head(object_hidden, triplet_indices)
        return SceneGraphHeadOutput(
            edge_logits=edge_logits,
            between_logits=between_logits,
            pair_indices=pair_indices,
            triplet_indices=triplet_indices,
        )


def pair_valid_mask(object_mask: torch.Tensor, pair_indices: torch.Tensor) -> torch.Tensor:
    return object_mask[:, pair_indices[:, 0]] & object_mask[:, pair_indices[:, 1]]


def triplet_valid_mask(object_mask: torch.Tensor, triplet_indices: torch.Tensor) -> torch.Tensor:
    return (
        object_mask[:, triplet_indices[:, 0]]
        & object_mask[:, triplet_indices[:, 1]]
        & object_mask[:, triplet_indices[:, 2]]
    )


def scene_graph_losses(
    output: SceneGraphHeadOutput,
    edge_labels: torch.Tensor,
    between_labels: torch.Tensor,
    object_mask: torch.Tensor,
    lambda_edge: float = 0.1,
    lambda_between: float = 0.1,
    edge_pos_weight: Optional[torch.Tensor] = None,
    between_pos_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    valid_pairs = pair_valid_mask(object_mask, output.pair_indices)
    valid_triplets = triplet_valid_mask(object_mask, output.triplet_indices)

    if valid_pairs.any():
        edge_loss = F.binary_cross_entropy_with_logits(
            output.edge_logits[valid_pairs],
            edge_labels.to(output.edge_logits.dtype)[valid_pairs],
            pos_weight=edge_pos_weight,
        )
    else:
        edge_loss = output.edge_logits.sum() * 0.0

    if valid_triplets.any():
        between_loss = F.binary_cross_entropy_with_logits(
            output.between_logits[valid_triplets],
            between_labels.to(output.between_logits.dtype)[valid_triplets],
            pos_weight=between_pos_weight,
        )
    else:
        between_loss = output.between_logits.sum() * 0.0

    return {
        "edge_loss": edge_loss,
        "between_loss": between_loss,
        "graph_loss": lambda_edge * edge_loss + lambda_between * between_loss,
    }


def multilabel_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, threshold: float = 0.0) -> Dict[str, float]:
    if not mask.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    preds = logits[mask] > threshold
    gold = labels[mask].bool()
    tp = (preds & gold).sum().item()
    fp = (preds & ~gold).sum().item()
    fn = (~preds & gold).sum().item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
