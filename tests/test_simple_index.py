from __future__ import annotations

import json
from pathlib import Path

import pytest

from industry_ml_lab.retrieval.simple_index import build_index, load_index, search_index


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_build_load_and_search_index_round_trip(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    index_path = tmp_path / "index.json"
    _write_jsonl(
        records_path,
        [
            {"id": "cat", "embedding": [1.0, 0.0], "metadata": {"label": "cat"}},
            {"id": "dog", "embedding": [0.0, 1.0], "metadata": {"label": "dog"}},
            {"id": "fox", "embedding": [0.8, 0.2], "metadata": {"label": "fox"}},
        ],
    )

    build_index(records_path, index_path)
    index = load_index(index_path)
    results = search_index(index, [0.95, 0.05], top_k=2)

    assert len(index) == 3
    assert [match["item_id"] for match in results] == ["cat", "fox"]
    assert results[0]["score"] > results[1]["score"]


def test_build_index_rejects_inconsistent_dimensions(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    _write_jsonl(
        records_path,
        [
            {"id": "cat", "embedding": [1.0, 0.0]},
            {"id": "bad", "embedding": [1.0, 0.0, 0.5]},
        ],
    )

    with pytest.raises(ValueError, match="Embedding dimensions are inconsistent"):
        build_index(records_path, tmp_path / "index.json")


def test_search_index_raises_for_dimension_mismatch(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    index_path = tmp_path / "index.json"
    _write_jsonl(
        records_path,
        [
            {"id": "cat", "embedding": [1.0, 0.0]},
            {"id": "dog", "embedding": [0.0, 1.0]},
        ],
    )

    build_index(records_path, index_path)
    index = load_index(index_path)

    with pytest.raises(ValueError):
        search_index(index, [1.0], top_k=1)

