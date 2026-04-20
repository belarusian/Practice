from __future__ import annotations

import csv
import json
from pathlib import Path

from industry_ml_lab.active_learning.scoring import score_predictions, write_relabel_queue


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_score_predictions_sorts_by_uncertainty(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "sample_id": "certain",
                "source_uri": "s3://bucket/certain.wav",
                "labels": ["cat", "dog", "bird"],
                "probabilities": [0.98, 0.01, 0.01],
            },
            {
                "sample_id": "uncertain",
                "source_uri": "s3://bucket/uncertain.wav",
                "labels": ["cat", "dog", "bird"],
                "probabilities": [0.34, 0.33, 0.33],
            },
        ],
    )

    scored = score_predictions(predictions_path)

    assert [item["sample_id"] for item in scored] == ["uncertain", "certain"]
    assert scored[0]["predicted_label"] == "cat"
    assert scored[0]["source_uri"] == "s3://bucket/uncertain.wav"
    assert scored[0]["uncertainty"] > scored[1]["uncertainty"]


def test_score_predictions_falls_back_to_index_for_missing_labels(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "sample_id": "no-labels",
                "probabilities": [0.1, 0.8, 0.1],
            }
        ],
    )

    scored = score_predictions(predictions_path)

    assert scored[0]["predicted_label"] == "1"
    assert scored[0]["predicted_score"] == 0.8


def test_write_relabel_queue_respects_limit(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "relabel-queue.csv"
    _write_jsonl(
        predictions_path,
        [
            {"sample_id": "high", "labels": ["a", "b"], "probabilities": [0.51, 0.49]},
            {"sample_id": "medium", "labels": ["a", "b"], "probabilities": [0.7, 0.3]},
            {"sample_id": "low", "labels": ["a", "b"], "probabilities": [0.99, 0.01]},
        ],
    )

    write_relabel_queue(predictions_path, output_path, limit=2)

    with output_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["sample_id"] for row in rows] == ["high", "medium"]
    assert rows[0]["predicted_label"] == "a"

