from __future__ import annotations

import json
from pathlib import Path

import pytest

from industry_ml_lab import cli


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_parse_vector_parses_csv_values() -> None:
    assert cli._parse_vector("0.9, 0.08,0.02") == [0.9, 0.08, 0.02]


def test_parse_vector_rejects_invalid_value() -> None:
    with pytest.raises(SystemExit, match="Invalid vector"):
        cli._parse_vector("0.9,not-a-number")


def test_build_index_command_writes_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "index.json"
    _write_jsonl(
        records_path,
        [
            {"id": "cat", "embedding": [1.0, 0.0], "metadata": {"label": "cat"}},
            {"id": "dog", "embedding": [0.0, 1.0], "metadata": {"label": "dog"}},
        ],
    )

    cli.main(["build-index", "--records", str(records_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    assert captured.out.strip() == str(output_path)
    assert output_path.exists()


def test_search_index_command_prints_ranked_matches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records_path = tmp_path / "records.jsonl"
    index_path = tmp_path / "index.json"
    _write_jsonl(
        records_path,
        [
            {"id": "cat", "embedding": [1.0, 0.0], "metadata": {"label": "cat"}},
            {"id": "dog", "embedding": [0.0, 1.0], "metadata": {"label": "dog"}},
        ],
    )
    cli.main(["build-index", "--records", str(records_path), "--output", str(index_path)])
    capsys.readouterr()

    cli.main(
        ["search-index", "--index-path", str(index_path), "--vector", "0.9,0.1", "--top-k", "1"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["item_id"] == "cat"


def test_build_text_index_command_writes_output_with_fake_embedder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from industry_ml_lab.retrieval import text_embeddings

    class FakeEmbedder:
        model_name = "fake-cli-encoder"

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] if "gpu" in text.lower() else [0.0, 1.0] for text in texts]

    monkeypatch.setattr(text_embeddings, "TransformerTextEmbedder", FakeEmbedder)

    records_path = tmp_path / "text-records.jsonl"
    output_path = tmp_path / "text-index.json"
    _write_jsonl(
        records_path,
        [
            {"id": "gpu", "text": "GPU training uses CUDA."},
            {"id": "audio", "text": "Audio training uses waveforms."},
        ],
    )

    cli.main(["build-text-index", "--records", str(records_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    assert captured.out.strip() == str(output_path)
    assert output_path.exists()


def test_search_text_index_command_prints_ranked_matches_with_fake_embedder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from industry_ml_lab.retrieval import text_embeddings

    class FakeEmbedder:
        model_name = "fake-cli-encoder"

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [
                [1.0, 0.0] if "gpu" in text.lower() or "cuda" in text.lower() else [0.0, 1.0]
                for text in texts
            ]

    monkeypatch.setattr(text_embeddings, "TransformerTextEmbedder", FakeEmbedder)

    records_path = tmp_path / "text-records.jsonl"
    index_path = tmp_path / "text-index.json"
    _write_jsonl(
        records_path,
        [
            {"id": "gpu", "text": "GPU training uses CUDA."},
            {"id": "audio", "text": "Audio training uses waveforms."},
        ],
    )
    cli.main(["build-text-index", "--records", str(records_path), "--output", str(index_path)])
    capsys.readouterr()

    cli.main(
        [
            "search-text-index",
            "--index-path",
            str(index_path),
            "--query",
            "CUDA memory",
            "--top-k",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["item_id"] == "gpu"


def test_active_learning_report_command_writes_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "queue.csv"
    _write_jsonl(
        predictions_path,
        [
            {"sample_id": "a", "labels": ["x", "y"], "probabilities": [0.5, 0.5]},
            {"sample_id": "b", "labels": ["x", "y"], "probabilities": [0.95, 0.05]},
        ],
    )

    cli.main(
        [
            "active-learning-report",
            "--predictions",
            str(predictions_path),
            "--output",
            str(output_path),
            "--limit",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert captured.out.strip() == str(output_path)
    assert output_path.exists()
    assert "sample_id" in output_path.read_text(encoding="utf-8")


def test_check_command_json_exits_nonzero_when_training_deps_are_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from industry_ml_lab.training import checklist

    dataset_root = tmp_path / "data" / "vision"
    dataset_root.mkdir(parents=True)

    monkeypatch.setattr(checklist, "_has_module", lambda name: False)

    with pytest.raises(SystemExit, match="1"):
        cli.main(
            [
                "check",
                "--target",
                "vision",
                "--device",
                "cpu",
                "--output-dir",
                str(tmp_path / "artifacts" / "vision"),
                "--dataset-root",
                str(dataset_root),
                "--json",
            ]
        )

    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "vision"
    assert payload["resolved_device"] == "cpu"
    assert payload["is_ready"] is False
    assert {check["name"] for check in payload["checks"] if check["status"] == "error"} == {
        "dependency_torch",
        "dependency_torchvision",
    }


def test_check_command_supports_text_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from industry_ml_lab.training import checklist

    dataset_root = tmp_path / "data" / "text"
    dataset_root.mkdir(parents=True)

    monkeypatch.setattr(checklist, "_has_module", lambda name: name in {"torch"})

    with pytest.raises(SystemExit, match="1"):
        cli.main(
            [
                "check",
                "--target",
                "text",
                "--device",
                "cpu",
                "--output-dir",
                str(tmp_path / "artifacts" / "text"),
                "--dataset-root",
                str(dataset_root),
                "--json",
            ]
        )

    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "text"
    assert payload["resolved_device"] == "cpu"
    assert payload["is_ready"] is False
    assert {check["name"] for check in payload["checks"] if check["status"] == "error"} == {
        "dependency_transformers",
        "dependency_datasets",
    }


def test_train_vision_fails_with_checklist_output_before_heavy_imports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from industry_ml_lab.training import checklist

    monkeypatch.setattr(checklist, "_has_module", lambda name: False)

    with pytest.raises(SystemExit, match="1"):
        cli.main(
            [
                "train-vision",
                "--output-dir",
                str(tmp_path / "artifacts" / "vision"),
            ]
        )

    output = capsys.readouterr().out
    assert "TRAINING ENVIRONMENT CHECKLIST" in output
    assert "dependency_torch" in output
    assert "ModuleNotFoundError" not in output


def test_train_text_classifier_fails_with_checklist_output_before_heavy_imports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from industry_ml_lab.training import checklist

    monkeypatch.setattr(checklist, "_has_module", lambda name: False)

    with pytest.raises(SystemExit, match="1"):
        cli.main(
            [
                "train-text-classifier",
                "--output-dir",
                str(tmp_path / "artifacts" / "text"),
            ]
        )

    output = capsys.readouterr().out
    assert "TRAINING ENVIRONMENT CHECKLIST" in output
    assert "dependency_transformers" in output
    assert "dependency_datasets" in output
    assert "ModuleNotFoundError" not in output
