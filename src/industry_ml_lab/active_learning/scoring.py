from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def _normalized_entropy(probabilities: list[float]) -> float:
    positive = [value for value in probabilities if value > 0]
    if len(positive) <= 1:
        return 0.0
    entropy = -sum(value * math.log(value) for value in positive)
    return entropy / math.log(len(probabilities))


def _margin(probabilities: list[float]) -> float:
    if len(probabilities) < 2:
        return 1.0
    ranked = sorted(probabilities, reverse=True)
    return ranked[0] - ranked[1]


def _uncertainty_score(probabilities: list[float]) -> float:
    entropy = _normalized_entropy(probabilities)
    margin = _margin(probabilities)
    return 0.7 * entropy + 0.3 * (1.0 - margin)


def _read_predictions(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def score_predictions(path: Path) -> list[dict[str, object]]:
    scored = []
    for record in _read_predictions(path):
        probabilities = [float(value) for value in record["probabilities"]]
        labels = record.get("labels") or []
        best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        scored.append(
            {
                "sample_id": record["sample_id"],
                "source_uri": record.get("source_uri", ""),
                "predicted_label": labels[best_index] if labels else str(best_index),
                "predicted_score": probabilities[best_index],
                "margin": _margin(probabilities),
                "entropy": _normalized_entropy(probabilities),
                "uncertainty": _uncertainty_score(probabilities),
            }
        )

    return sorted(scored, key=lambda item: item["uncertainty"], reverse=True)


def write_relabel_queue(predictions_path: Path, output_path: Path, limit: int = 100) -> None:
    ranked = score_predictions(predictions_path)[:limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "source_uri",
                "predicted_label",
                "predicted_score",
                "margin",
                "entropy",
                "uncertainty",
            ],
        )
        writer.writeheader()
        writer.writerows(ranked)

