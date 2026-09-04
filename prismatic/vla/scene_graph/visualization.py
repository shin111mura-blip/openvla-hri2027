"""Visualization helpers for BBox-derived scene graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageDraw, ImageFont

from prismatic.vla.scene_graph.schema import SceneGraph


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def draw_bbox_overlay(image: Image.Image, scene_graph: SceneGraph) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    font = _font()
    for node in scene_graph.nodes:
        x1, y1, x2, y2 = node.bbox
        box = (x1 * w, y1 * h, x2 * w, y2 * h)
        draw.rectangle(box, outline=(255, 80, 0), width=2)
        label = f"{node.node_id}:{node.category}[{node.instance_id}] {node.confidence:.2f}"
        draw.text((box[0] + 2, box[1] + 2), label, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return out


def draw_pairwise_overlay(image: Image.Image, scene_graph: SceneGraph, relation_filter: Iterable[str] | None = None) -> Image.Image:
    out = draw_bbox_overlay(image, scene_graph)
    draw = ImageDraw.Draw(out)
    w, h = out.size
    keep = set(relation_filter) if relation_filter is not None else None
    nodes = {node.node_id: node for node in scene_graph.nodes}
    for edge in scene_graph.pairwise_edges:
        rels = [rel for rel in edge.relations if keep is None or rel in keep]
        if not rels:
            continue
        src, dst = nodes[edge.source], nodes[edge.target]
        p1 = (src.center[0] * w, src.center[1] * h)
        p2 = (dst.center[0] * w, dst.center[1] * h)
        draw.line((p1, p2), fill=(0, 180, 255), width=2)
        mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
        draw.text((mx, my), ",".join(rels), fill=(0, 0, 0))
    return out


def draw_between_overlay(image: Image.Image, scene_graph: SceneGraph) -> Image.Image:
    out = draw_bbox_overlay(image, scene_graph)
    draw = ImageDraw.Draw(out)
    w, h = out.size
    nodes = {node.node_id: node for node in scene_graph.nodes}
    for edge in scene_graph.between_hyperedges:
        target = nodes[edge.target]
        ref1 = nodes[edge.reference_1]
        ref2 = nodes[edge.reference_2]
        points: Tuple[Tuple[float, float], ...] = (
            (ref1.center[0] * w, ref1.center[1] * h),
            (target.center[0] * w, target.center[1] * h),
            (ref2.center[0] * w, ref2.center[1] * h),
        )
        draw.line(points, fill=(255, 0, 180), width=3)
        draw.text(points[1], f"between({edge.target},{edge.reference_1},{edge.reference_2})", fill=(255, 255, 255))
    return out


def save_scene_graph_overlays(image: Image.Image, scene_graph: SceneGraph, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = scene_graph.image_id.replace("/", "__")
    draw_bbox_overlay(image, scene_graph).save(output_dir / f"{safe_id}_bbox.png")
    draw_pairwise_overlay(image, scene_graph).save(output_dir / f"{safe_id}_pairwise.png")
    draw_between_overlay(image, scene_graph).save(output_dir / f"{safe_id}_between.png")
