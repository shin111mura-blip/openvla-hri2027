"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

import json
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from itertools import combinations
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100
PAIRWISE_RELATIONS = ("left_of", "right_of", "above", "below", "near", "overlap")


def _scalar_int(value: Any) -> int:
    array = np.asarray(value)
    return int(array.reshape(-1)[0])


def _cache_key_from_image_id(image_id: str) -> Tuple[int, int]:
    demo = re.search(r"/demo_(\d+)/", image_id)
    step = re.search(r"/step_(\d+)/", image_id)
    if demo is None or step is None:
        raise ValueError(f"Cannot parse demo/timestep from image_id={image_id!r}")
    return int(demo.group(1)), int(step.group(1))


class BBoxSceneGraphCache:
    """Loads offline YOLO BBox and BBox-derived scene graph caches for RLDS samples."""

    def __init__(
        self,
        bbox_cache_path: Path,
        scene_graph_cache_path: Optional[Path],
        max_objects: int,
        require: bool,
    ) -> None:
        self.max_objects = max_objects
        self.require = require
        self.num_pairs = max_objects * (max_objects - 1)
        self.num_triplets = max_objects * ((max_objects - 1) * (max_objects - 2) // 2)
        self.bbox_by_key = self._load_bbox_cache(bbox_cache_path)
        self.graph_by_key = self._load_scene_graph_cache(scene_graph_cache_path) if scene_graph_cache_path else {}
        self.bbox_timesteps_by_demo: Dict[int, list[int]] = {}
        for demo_id, timestep in self.bbox_by_key:
            self.bbox_timesteps_by_demo.setdefault(demo_id, []).append(timestep)
        for timesteps in self.bbox_timesteps_by_demo.values():
            timesteps.sort()
        self._warned_fallback_keys: set[Tuple[int, int]] = set()

    @classmethod
    def from_env(cls) -> Optional["BBoxSceneGraphCache"]:
        bbox_cache = os.environ.get("OPENVLA_BBOX_CACHE_JSONL")
        scene_graph_cache = os.environ.get("OPENVLA_SCENE_GRAPH_CACHE_JSONL")
        if not bbox_cache and not scene_graph_cache:
            return None
        if not bbox_cache:
            raise ValueError("OPENVLA_SCENE_GRAPH_CACHE_JSONL requires OPENVLA_BBOX_CACHE_JSONL.")
        return cls(
            bbox_cache_path=Path(bbox_cache),
            scene_graph_cache_path=Path(scene_graph_cache) if scene_graph_cache else None,
            max_objects=int(os.environ.get("OPENVLA_MAX_OBJECTS", "12")),
            require=os.environ.get("OPENVLA_REQUIRE_BBOX_SCENE_GRAPH", "0").lower() in {"1", "true", "yes"},
        )

    def _load_bbox_cache(self, path: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
        records: Dict[Tuple[int, int], Dict[str, Any]] = {}
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                key = _cache_key_from_image_id(row["image_id"])
                detections = list(row.get("detections", []))[: self.max_objects]
                records[key] = {"image_id": row["image_id"], "detections": detections}
        return records

    def _load_scene_graph_cache(self, path: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
        records: Dict[Tuple[int, int], Dict[str, Any]] = {}
        relation_to_idx = {name: idx for idx, name in enumerate(PAIRWISE_RELATIONS)}
        pair_to_idx = {
            (i, j): idx
            for idx, (i, j) in enumerate(
                (i, j) for i in range(self.max_objects) for j in range(self.max_objects) if i != j
            )
        }
        triplet_to_idx = {
            (target, ref1, ref2): idx
            for idx, (target, ref1, ref2) in enumerate(
                (target, ref1, ref2)
                for target in range(self.max_objects)
                for ref1, ref2 in combinations([idx for idx in range(self.max_objects) if idx != target], 2)
            )
        }
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                edge_labels = np.zeros((self.num_pairs, len(PAIRWISE_RELATIONS)), dtype=np.float32)
                between_labels = np.zeros((self.num_triplets,), dtype=np.float32)
                for edge in row.get("pairwise_edges", []):
                    pair_idx = pair_to_idx.get((int(edge["source"]), int(edge["target"])))
                    if pair_idx is None:
                        continue
                    for relation in edge.get("relations", []):
                        relation_idx = relation_to_idx.get(relation)
                        if relation_idx is not None:
                            edge_labels[pair_idx, relation_idx] = 1.0
                for edge in row.get("between_hyperedges", []):
                    ref1, ref2 = sorted((int(edge["reference_1"]), int(edge["reference_2"])))
                    triplet_idx = triplet_to_idx.get((int(edge["target"]), ref1, ref2))
                    if triplet_idx is not None:
                        between_labels[triplet_idx] = 1.0
                records[_cache_key_from_image_id(row["image_id"])] = {
                    "edge_labels": edge_labels,
                    "between_labels": between_labels,
                }
        return records

    def get(self, demo_id: int, timestep: int) -> Dict[str, Any]:
        key = (demo_id, timestep)
        bbox_record = self.bbox_by_key.get(key)
        graph_key = key
        if bbox_record is None:
            fallback_key = self._fallback_key(demo_id, timestep)
            if fallback_key is not None:
                bbox_record = self.bbox_by_key[fallback_key]
                graph_key = fallback_key
                if key not in self._warned_fallback_keys:
                    warnings.warn(
                        "Missing exact BBox cache entry for "
                        f"demo_id={demo_id}, timestep={timestep}; using timestep={fallback_key[1]} "
                        "from the same demo. Regenerate the cache from the exact training stream "
                        "to remove this fallback.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_fallback_keys.add(key)
            elif self.require:
                raise KeyError(f"Missing BBox cache entry for demo_id={demo_id}, timestep={timestep}")
        if bbox_record is None:
            bbox_record = {"image_id": f"missing/demo_{demo_id:03d}/step_{timestep:06d}", "detections": []}

        bboxes = np.zeros((self.max_objects, 4), dtype=np.float32)
        object_mask = np.zeros((self.max_objects,), dtype=np.bool_)
        confidences = np.zeros((self.max_objects,), dtype=np.float32)
        for idx, detection in enumerate(bbox_record["detections"][: self.max_objects]):
            bboxes[idx] = np.asarray(detection.get("bbox_normalized", [0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
            object_mask[idx] = True
            confidences[idx] = float(detection.get("confidence", 0.0))

        graph_record = self.graph_by_key.get(graph_key)
        if graph_record is None:
            if self.require and self.graph_by_key:
                raise KeyError(f"Missing Scene Graph cache entry for demo_id={demo_id}, timestep={timestep}")
            graph_record = {
                "edge_labels": np.zeros((self.num_pairs, len(PAIRWISE_RELATIONS)), dtype=np.float32),
                "between_labels": np.zeros((self.num_triplets,), dtype=np.float32),
            }

        return {
            "image_id": bbox_record["image_id"],
            "bboxes_normalized": torch.from_numpy(bboxes),
            "object_mask": torch.from_numpy(object_mask),
            "confidences": torch.from_numpy(confidences),
            "edge_labels": torch.from_numpy(graph_record["edge_labels"]),
            "between_labels": torch.from_numpy(graph_record["between_labels"]),
        }

    def _fallback_key(self, demo_id: int, timestep: int) -> Optional[Tuple[int, int]]:
        timesteps = self.bbox_timesteps_by_demo.get(demo_id)
        if not timesteps:
            return None
        previous = [ts for ts in timesteps if ts <= timestep]
        if previous:
            return (demo_id, previous[-1])
        return (demo_id, timesteps[0])


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __post_init__(self) -> None:
        self.bbox_scene_graph_cache = BBoxSceneGraphCache.from_env()
        self.graph_internalization_sidecar = None
        if os.environ.get("OPENVLA_GRAPH_INTERNALIZATION", "").lower() in {"1", "true", "yes"}:
            from graph_internalization.rlds_hook import RLDSSidecarLookup

            self.graph_internalization_sidecar = RLDSSidecarLookup.from_env()

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        output = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)
        if self.bbox_scene_graph_cache is not None:
            if "episode_index" not in rlds_batch["observation"]:
                raise KeyError(
                    "BBox/Scene Graph cache was requested but RLDS sample has no observation['episode_index']. "
                    "Set OPENVLA_RLDS_DEMO_SPLIT_JSON so the local dataset patch can preserve episode IDs."
                )
            demo_id = _scalar_int(rlds_batch["observation"]["episode_index"][0])
            timestep = _scalar_int(rlds_batch["observation"]["timestep"][0])
            output.update(self.bbox_scene_graph_cache.get(demo_id=demo_id, timestep=timestep))
            output.update({"demo_id": demo_id, "timestep": timestep})
        if self.graph_internalization_sidecar is not None:
            output.update(self.graph_internalization_sidecar.lookup(rlds_batch))
        return output


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
