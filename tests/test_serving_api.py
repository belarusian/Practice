from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

HAS_SERVING_TEST_DEPS = (
    importlib.util.find_spec("fastapi") is not None and importlib.util.find_spec("httpx") is not None
)

pytestmark = pytest.mark.skipif(
    not HAS_SERVING_TEST_DEPS,
    reason="serving extra not installed",
)

if HAS_SERVING_TEST_DEPS:
    from fastapi.testclient import TestClient

    from industry_ml_lab.retrieval.simple_index import build_index
    from industry_ml_lab.serving import api


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def clear_api_caches() -> None:
    api.get_settings.cache_clear()
    api.get_embedding_index.cache_clear()
    api.get_predictor.cache_clear()
    yield
    api.get_settings.cache_clear()
    api.get_embedding_index.cache_clear()
    api.get_predictor.cache_clear()


def test_healthz_reports_service_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_LAB_SERVICE_VERSION", "test-version")

    client = TestClient(api.app)
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "industry-ml-lab-api",
        "version": "test-version",
    }


def test_readyz_reports_service_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_LAB_SERVICE_VERSION", "ready-version")

    client = TestClient(api.app)
    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service": "industry-ml-lab-api",
        "version": "ready-version",
    }


def test_embedding_search_returns_ranked_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    monkeypatch.setenv("ML_LAB_INDEX_PATH", str(index_path))

    client = TestClient(api.app)
    response = client.post(
        "/embeddings/search",
        json={"vector": [0.95, 0.05], "top_k": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [match["item_id"] for match in payload["matches"]] == ["cat", "fox"]
    assert payload["matches"][0]["metadata"] == {"label": "cat"}
    assert payload["matches"][0]["score"] > payload["matches"][1]["score"]


def test_embedding_search_returns_503_when_index_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_index = tmp_path / "missing-index.json"
    monkeypatch.setenv("ML_LAB_INDEX_PATH", str(missing_index))

    client = TestClient(api.app)
    response = client.post(
        "/embeddings/search",
        json={"vector": [0.95, 0.05], "top_k": 2},
    )

    assert response.status_code == 503
    assert str(missing_index) in response.json()["detail"]
