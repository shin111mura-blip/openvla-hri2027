"""Local logging utilities for BBox/scene graph experiments."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch


TRAIN_METRIC_FIELDS = [
    "global_step",
    "epoch",
    "wall_clock_time",
    "samples_seen",
    "optimizer_updates",
    "learning_rate",
    "micro_action_loss",
    "micro_edge_loss",
    "micro_between_loss",
    "update_action_loss",
    "update_edge_loss",
    "update_between_loss",
    "update_total_loss",
    "gradient_norm_total",
    "gradient_norm_lora",
    "gradient_norm_bbox_token_encoder",
    "gradient_norm_edge_head",
    "gradient_norm_between_head",
    "gpu_memory_allocated",
    "gpu_memory_reserved",
    "step_time",
    "data_loading_time",
    "image_id",
    "task_id",
    "demo_id",
    "timestep",
]


class LocalMetricLogger:
    def __init__(self, run_dir: Path, fieldnames: Iterable[str] = TRAIN_METRIC_FIELDS) -> None:
        self.run_dir = Path(run_dir)
        self.log_dir = self.run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "train_metrics.jsonl"
        self.csv_path = self.log_dir / "train_metrics.csv"
        self.fieldnames = list(fieldnames)
        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def log(self, row: Mapping[str, Any]) -> None:
        clean = {key: _jsonable(row.get(key, "")) for key in self.fieldnames}
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(clean) + "\n")
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(clean)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    norms = [torch.linalg.vector_norm(p.grad.detach(), 2) for p in parameters if p.grad is not None]
    if not norms:
        return 0.0
    return torch.linalg.vector_norm(torch.stack(norms), 2).item()


def module_grad_norm(module: torch.nn.Module | None) -> float:
    if module is None:
        return 0.0
    return grad_norm(module.parameters())


def cuda_memory() -> Dict[str, int]:
    if not torch.cuda.is_available():
        return {"gpu_memory_allocated": 0, "gpu_memory_reserved": 0}
    return {
        "gpu_memory_allocated": int(torch.cuda.memory_allocated()),
        "gpu_memory_reserved": int(torch.cuda.memory_reserved()),
    }


def write_run_metadata(run_dir: Path, config: Mapping[str, Any], command_line: str, exit_status: str = "prepared") -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_name": run_dir.name,
        "created_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
        "resolved_config": dict(config),
        "command_line": command_line,
        "exit_status": exit_status,
    }
    try:
        metadata["openvla_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        metadata["openvla_status"] = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    except Exception as exc:  # pragma: no cover - metadata best effort
        metadata["git_error"] = repr(exc)
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
