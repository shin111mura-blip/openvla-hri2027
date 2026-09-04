"""Build scene graph supervision from YOLO-style BBox detections."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import Any, Dict, Sequence

from prismatic.vla.scene_graph.relations import RelationThresholds, is_between, pairwise_relations
from prismatic.vla.scene_graph.schema import (
    BetweenHyperedge,
    PairwiseEdge,
    SceneGraph,
    SceneGraphNode,
    assign_instance_ids,
    sort_detections_spatial,
)


def build_scene_graph(
    image_id: str,
    detections: Sequence[Dict[str, Any]],
    thresholds: RelationThresholds | Dict[str, float] | None = None,
    max_objects: int | None = None,
    source_bbox_cache_id: str = "",
) -> SceneGraph:
    threshold_obj = thresholds if isinstance(thresholds, RelationThresholds) else RelationThresholds.from_dict(thresholds)
    ordered = sort_detections_spatial(detections, max_objects=max_objects)
    instance_ids = assign_instance_ids(ordered)
    nodes = tuple(
        SceneGraphNode.from_detection(node_id=i, detection=det, instance_id=instance_ids[i])
        for i, det in enumerate(ordered)
    )

    pairwise_edges = []
    for src in nodes:
        for dst in nodes:
            rels = pairwise_relations(src, dst, threshold_obj)
            if rels:
                pairwise_edges.append(PairwiseEdge(source=src.node_id, target=dst.node_id, relations=rels))

    between_edges = set()
    for target in nodes:
        for ref1, ref2 in combinations(nodes, 2):
            if is_between(target, ref1, ref2, threshold_obj):
                between_edges.add(BetweenHyperedge(target.node_id, ref1.node_id, ref2.node_id).canonical())

    return SceneGraph(
        image_id=image_id,
        nodes=nodes,
        pairwise_edges=tuple(pairwise_edges),
        between_hyperedges=tuple(sorted(between_edges, key=lambda e: (e.target, e.reference_1, e.reference_2))),
        threshold_config=threshold_obj.to_dict(),
        source_bbox_cache_id=source_bbox_cache_id,
    )


def scene_graph_hash(scene_graph: SceneGraph) -> str:
    payload = json.dumps(scene_graph.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
