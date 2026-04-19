from __future__ import annotations

import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from industry_ml_lab.config import AppSettings
from industry_ml_lab.workflows.activities import generate_relabel_queue, run_vision_training
from industry_ml_lab.workflows.workflows import VisionTrainingWorkflow


async def _run(task_queue: str | None = None) -> None:
    settings = AppSettings.from_env()
    client = await Client.connect(settings.temporal_address)
    worker = Worker(
        client,
        task_queue=task_queue or settings.temporal_task_queue,
        workflows=[VisionTrainingWorkflow],
        activities=[run_vision_training, generate_relabel_queue],
    )
    await worker.run()


def run_worker(task_queue: str | None = None) -> None:
    asyncio.run(_run(task_queue=task_queue))

