from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class IndexedRecord:
    item_id: str
    vector: list[float]
    norm: float
    metadata: dict[str, Any]


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _dot(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


def _cosine_similarity(left: list[float], right: list[float], right_norm: float) -> float:
    left_norm = _vector_norm(left)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return _dot(left, right) / (left_norm * right_norm)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def build_index(records_path: Path, output_path: Path) -> None:
    records = _read_jsonl(records_path)
    if not records:
        raise ValueError(f"No records found in {records_path}")

    width = len(records[0]["embedding"])
    indexed = []
    for record in records:
        vector = [float(value) for value in record["embedding"]]
        if len(vector) != width:
            raise ValueError("Embedding dimensions are inconsistent in the source file.")
        indexed.append(
            {
                "item_id": record["id"],
                "vector": vector,
                "norm": _vector_norm(vector),
                "metadata": record.get("metadata", {}),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"dimension": width, "records": indexed}, indent=2), encoding="utf-8")


def load_index(index_path: Path) -> list[IndexedRecord]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return [
        IndexedRecord(
            item_id=record["item_id"],
            vector=[float(value) for value in record["vector"]],
            norm=float(record["norm"]),
            metadata=record.get("metadata", {}),
        )
        for record in payload["records"]
    ]


def search_index(index: list[IndexedRecord], query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
    matches = []
    for record in index:
        score = _cosine_similarity(query_vector, record.vector, record.norm)
        matches.append(
            {
                "item_id": record.item_id,
                "score": score,
                "metadata": record.metadata,
            }
        )

    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[:top_k]

