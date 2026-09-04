"""Typed scene graph schema for BBox-derived 2D relation labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple


PAIRWISE_RELATIONS = ("left_of", "right_of", "above", "below", "near", "overlap")
SCENE_GRAPH_SCHEMA_VERSION = "bbox2d-v1"


@dataclass(frozen=True)
class SceneGraphNode:
    node_id: int
    instance_id: int
    category: str
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    width: float
    height: float
    confidence: float
    valid: bool = True

    @classmethod
    def from_detection(cls, node_id: int, detection: Dict[str, Any], instance_id: int) -> "SceneGraphNode":
        bbox = tuple(float(x) for x in detection.get("bbox_normalized", detection.get("bbox_xyxy", (0, 0, 0, 0))))
        x1, y1, x2, y2 = bbox
        return cls(
            node_id=node_id,
            instance_id=instance_id,
            category=str(detection.get("category", "object")),
            bbox=bbox,
            center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            width=max(0.0, x2 - x1),
            height=max(0.0, y2 - y1),
            confidence=float(detection.get("confidence", 1.0)),
            valid=bool(detection.get("valid", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairwiseEdge:
    source: int
    target: int
    relations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "target": self.target, "relations": list(self.relations)}


@dataclass(frozen=True)
class BetweenHyperedge:
    target: int
    reference_1: int
    reference_2: int

    def canonical(self) -> "BetweenHyperedge":
        r1, r2 = sorted((self.reference_1, self.reference_2))
        return BetweenHyperedge(target=self.target, reference_1=r1, reference_2=r2)

    def to_dict(self) -> Dict[str, Any]:
        edge = self.canonical()
        return {"target": edge.target, "reference_1": edge.reference_1, "reference_2": edge.reference_2}


@dataclass(frozen=True)
class SceneGraph:
    image_id: str
    nodes: Tuple[SceneGraphNode, ...]
    pairwise_edges: Tuple[PairwiseEdge, ...]
    between_hyperedges: Tuple[BetweenHyperedge, ...]
    threshold_config: Dict[str, float]
    source_bbox_cache_id: str = ""
    schema_version: str = SCENE_GRAPH_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "pairwise_edges": [edge.to_dict() for edge in self.pairwise_edges],
            "between_hyperedges": [edge.to_dict() for edge in self.between_hyperedges],
            "relation_schema_version": self.schema_version,
            "threshold_config": dict(self.threshold_config),
            "source_bbox_cache_id": self.source_bbox_cache_id,
        }


def sort_detections_spatial(detections: Sequence[Dict[str, Any]], max_objects: int | None = None) -> List[Dict[str, Any]]:
    """Sort detections left-to-right, then top-to-bottom, preserving no target/reference hints."""

    def key(det: Dict[str, Any]) -> Tuple[float, float]:
        bbox = det.get("bbox_normalized", det.get("bbox_xyxy", (0, 0, 0, 0)))
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    ordered = sorted(detections, key=key)
    if max_objects is not None:
        ordered = ordered[:max_objects]
    return ordered


def assign_instance_ids(detections: Sequence[Dict[str, Any]]) -> List[int]:
    counts: Dict[str, int] = {}
    ids: List[int] = []
    for det in detections:
        category = str(det.get("category", "object"))
        ids.append(counts.get(category, 0))
        counts[category] = counts.get(category, 0) + 1
    return ids
