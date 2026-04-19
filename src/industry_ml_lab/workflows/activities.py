from __future__ import annotations

import asyncio
from pathlib import Path

from temporalio import activity

from industry_ml_lab.active_learning.scoring import write_relabel_queue
from industry_ml_lab.config import VisionTrainConfig
from industry_ml_lab.training.vision import train
from industry_ml_lab.workflows.types import ModelArtifact, VisionTrainingRequest


@activity.defn
async def run_vision_training(request: VisionTrainingRequest) -> ModelArtifact:
    config = VisionTrainConfig(
        dataset_root=Path(request.dataset_root),
        output_dir=Path(request.output_dir),
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        device=request.device,
    )
    await asyncio.to_thread(train, config)
    return ModelArtifact(
        artifact_path=str(config.output_dir / "model.pt"),
        metrics_path=str(config.output_dir / "metrics.json"),
        model_name="vision-baseline",
    )


@activity.defn
async def generate_relabel_queue(predictions_path: str, output_path: str, limit: int = 100) -> str:
    await asyncio.to_thread(write_relabel_queue, Path(predictions_path), Path(output_path), limit)
    return output_path

