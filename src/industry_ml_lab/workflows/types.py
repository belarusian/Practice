from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VisionTrainingRequest:
    dataset_root: str
    output_dir: str
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 3e-4
    device: str | None = None


@dataclass
class ModelArtifact:
    artifact_path: str
    metrics_path: str
    model_name: str

