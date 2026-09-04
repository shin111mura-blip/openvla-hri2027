"""2D relation predicates over normalized bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, Tuple

from prismatic.vla.scene_graph.schema import PAIRWISE_RELATIONS, SceneGraphNode


@dataclass(frozen=True)
class RelationThresholds:
    near_threshold: float = 0.25
    left_right_margin: float = 0.02
    above_below_margin: float = 0.02
    overlap_iou_threshold: float = 0.05
    between_perpendicular_threshold: float = 0.10
    between_endpoint_margin: float = 0.05

    @classmethod
    def from_dict(cls, data: Dict[str, float] | None) -> "RelationThresholds":
        if data is None:
            return cls()
        valid = {field: float(data[field]) for field in cls.__dataclass_fields__ if field in data}
        return cls(**valid)

    def to_dict(self) -> Dict[str, float]:
        return {
            "near_threshold": self.near_threshold,
            "left_right_margin": self.left_right_margin,
            "above_below_margin": self.above_below_margin,
            "overlap_iou_threshold": self.overlap_iou_threshold,
            "between_perpendicular_threshold": self.between_perpendicular_threshold,
            "between_endpoint_margin": self.between_endpoint_margin,
        }


def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else inter / denom


def pairwise_relations(a: SceneGraphNode, b: SceneGraphNode, thresholds: RelationThresholds) -> Tuple[str, ...]:
    if not a.valid or not b.valid or a.node_id == b.node_id:
        return ()

    relations = []
    ax, ay = a.center
    bx, by = b.center
    if ax < bx - thresholds.left_right_margin:
        relations.append("left_of")
    if ax > bx + thresholds.left_right_margin:
        relations.append("right_of")
    if ay < by - thresholds.above_below_margin:
        relations.append("above")
    if ay > by + thresholds.above_below_margin:
        relations.append("below")
    if sqrt((ax - bx) ** 2 + (ay - by) ** 2) <= thresholds.near_threshold:
        relations.append("near")
    if bbox_iou(a.bbox, b.bbox) >= thresholds.overlap_iou_threshold:
        relations.append("overlap")
    return tuple(rel for rel in relations if rel in PAIRWISE_RELATIONS)


def is_between(
    target: SceneGraphNode,
    reference_1: SceneGraphNode,
    reference_2: SceneGraphNode,
    thresholds: RelationThresholds,
) -> bool:
    """Return whether target center lies near the line segment between two reference centers."""

    if len({target.node_id, reference_1.node_id, reference_2.node_id}) != 3:
        return False
    if not (target.valid and reference_1.valid and reference_2.valid):
        return False

    tx, ty = target.center
    ax, ay = reference_1.center
    bx, by = reference_2.center
    vx, vy = bx - ax, by - ay
    wx, wy = tx - ax, ty - ay
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq <= 1e-12:
        return False

    t = (wx * vx + wy * vy) / seg_len_sq
    if t < -thresholds.between_endpoint_margin or t > 1.0 + thresholds.between_endpoint_margin:
        return False

    proj_x, proj_y = ax + t * vx, ay + t * vy
    perpendicular = sqrt((tx - proj_x) ** 2 + (ty - proj_y) ** 2)
    return perpendicular <= thresholds.between_perpendicular_threshold


def relation_vector(relations: Iterable[str]) -> Tuple[int, ...]:
    present = set(relations)
    return tuple(1 if rel in present else 0 for rel in PAIRWISE_RELATIONS)
