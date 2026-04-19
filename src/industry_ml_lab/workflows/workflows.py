from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

from industry_ml_lab.workflows.activities import generate_relabel_queue, run_vision_training
from industry_ml_lab.workflows.types import ModelArtifact, VisionTrainingRequest


@workflow.defn
class VisionTrainingWorkflow:
    @workflow.run
    async def run(
        self,
        request: VisionTrainingRequest,
        predictions_path: str | None = None,
    ) -> dict[str, str]:
        artifact: ModelArtifact = await workflow.execute_activity(
            run_vision_training,
            request,
            start_to_close_timeout=timedelta(hours=6),
        )

        response = {
            "artifact_path": artifact.artifact_path,
            "metrics_path": artifact.metrics_path,
            "model_name": artifact.model_name,
        }

        if predictions_path:
            queue_path = f"{request.output_dir}/relabel-queue.csv"
            relabel_queue = await workflow.execute_activity(
                generate_relabel_queue,
                args=[predictions_path, queue_path, 100],
                start_to_close_timeout=timedelta(minutes=10),
            )
            response["relabel_queue_path"] = relabel_queue

        return response

