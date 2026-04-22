from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from industry_ml_lab.retrieval.simple_index import load_index
from industry_ml_lab.retrieval.text_embeddings import build_text_index, search_text_index


class FakeEmbedder:
    model_name = "fake-text-encoder"

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [_fake_vector(text) for text in texts]


class BadCountEmbedder:
    model_name = "bad-count"

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0]]


def _fake_vector(text: str) -> list[float]:
    normalized = text.lower()
    if "gpu" in normalized or "cuda" in normalized or "training" in normalized:
        return [1.0, 0.0, 0.0]
    if "audio" in normalized or "speech" in normalized:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_build_text_index_writes_vectors_and_preserves_text_metadata(tmp_path: Path) -> None:
    records_path = tmp_path / "text-records.jsonl"
    index_path = tmp_path / "text-index.json"
    _write_jsonl(
        records_path,
        [
            {
                "id": "gpu",
                "text": "CUDA GPU training uses ephemeral workers.",
                "metadata": {"topic": "training"},
            },
            {
                "id": "audio",
                "text": "Speech Commands gives us an audio classification smoke test.",
                "metadata": {"topic": "audio"},
            },
        ],
    )

    build_text_index(records_path, index_path, embedder=FakeEmbedder())

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    index = load_index(index_path)
    assert payload["embedding_model"] == "fake-text-encoder"
    assert payload["dimension"] == 3
    assert [record.item_id for record in index] == ["gpu", "audio"]
    assert index[0].metadata["topic"] == "training"
    assert index[0].metadata["text"] == "CUDA GPU training uses ephemeral workers."


def test_search_text_index_embeds_query_and_returns_ranked_matches(tmp_path: Path) -> None:
    records_path = tmp_path / "text-records.jsonl"
    index_path = tmp_path / "text-index.json"
    _write_jsonl(
        records_path,
        [
            {"id": "serving", "text": "FastAPI serving exposes model health checks."},
            {"id": "audio", "text": "Audio classification trains on speech commands."},
            {"id": "gpu", "text": "CUDA GPU training needs free VRAM."},
        ],
    )
    build_text_index(records_path, index_path, embedder=FakeEmbedder())

    results = search_text_index(
        index_path, "free CUDA memory for training", embedder=FakeEmbedder(), top_k=2
    )

    assert [match["item_id"] for match in results] == ["gpu", "serving"]
    assert results[0]["score"] > results[1]["score"]


def test_build_text_index_rejects_missing_text_field(tmp_path: Path) -> None:
    records_path = tmp_path / "text-records.jsonl"
    _write_jsonl(records_path, [{"id": "bad"}])

    with pytest.raises(ValueError, match="must contain 'id' and 'text'"):
        build_text_index(records_path, tmp_path / "text-index.json", embedder=FakeEmbedder())


def test_build_text_index_rejects_embedder_count_mismatch(tmp_path: Path) -> None:
    records_path = tmp_path / "text-records.jsonl"
    _write_jsonl(
        records_path,
        [
            {"id": "one", "text": "first record"},
            {"id": "two", "text": "second record"},
        ],
    )

    with pytest.raises(ValueError, match="different number of vectors"):
        build_text_index(records_path, tmp_path / "text-index.json", embedder=BadCountEmbedder())


def test_search_text_index_rejects_empty_query(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Query must not be empty"):
        search_text_index(tmp_path / "missing.json", "   ", embedder=FakeEmbedder())
