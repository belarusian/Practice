from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return min(cpu_count, 8)


@dataclass(slots=True)
class VisionTrainConfig:
    dataset_root: Path
    output_dir: Path
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = _default_workers()
    pretrained: bool = True
    device: str | None = None


@dataclass(slots=True)
class AudioTrainConfig:
    dataset_root: Path
    output_dir: Path
    epochs: int = 3
    batch_size: int = 128
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    num_workers: int = _default_workers()
    device: str | None = None
    train_sample_limit: int | None = None
    val_sample_limit: int | None = None


@dataclass(slots=True)
class TextTrainConfig:
    dataset_root: Path
    output_dir: Path
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_workers: int = 0
    device: str | None = None
    model_name: str = "distilbert/distilbert-base-uncased"
    dataset_name: str = "glue"
    dataset_config: str | None = "sst2"
    text_column: str = "sentence"
    label_column: str = "label"
    max_length: int = 128
    train_sample_limit: int | None = None
    val_sample_limit: int | None = None


@dataclass(slots=True)
class AppSettings:
    artifact_root: Path
    model_path: Path
    index_path: Path
    service_version: str
    temporal_address: str
    temporal_task_queue: str

    @classmethod
    def from_env(cls) -> "AppSettings":
        artifact_root = Path(os.getenv("ML_LAB_ARTIFACT_ROOT", "artifacts"))
        return cls(
            artifact_root=artifact_root,
            model_path=Path(os.getenv("ML_LAB_MODEL_PATH", artifact_root / "vision-baseline/model.pt")),
            index_path=Path(os.getenv("ML_LAB_INDEX_PATH", artifact_root / "demo-index.json")),
            service_version=os.getenv("ML_LAB_SERVICE_VERSION", "0.1.0"),
            temporal_address=os.getenv("TEMPORAL_ADDRESS", "localhost:7233"),
            temporal_task_queue=os.getenv("TEMPORAL_TASK_QUEUE", "industry-ml-lab"),
        )
