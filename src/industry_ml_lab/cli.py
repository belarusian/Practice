from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from industry_ml_lab.config import AudioTrainConfig, TextTrainConfig, VisionTrainConfig


def _add_common_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-lab",
        description="Training and deployment playground for production-style ML systems.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    vision_parser = subparsers.add_parser(
        "train-vision", help="Train the image classification baseline."
    )
    _add_common_training_args(vision_parser)
    vision_parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)

    audio_parser = subparsers.add_parser(
        "train-audio", help="Train the audio classification baseline."
    )
    _add_common_training_args(audio_parser)

    text_parser = subparsers.add_parser(
        "train-text-classifier",
        help="Fine-tune a transformer text classification baseline.",
    )
    _add_common_training_args(text_parser)
    text_parser.add_argument("--model-name", default="distilbert/distilbert-base-uncased")
    text_parser.add_argument("--dataset-name", default="glue")
    text_parser.add_argument("--dataset-config", default="sst2")
    text_parser.add_argument("--text-column", default="sentence")
    text_parser.add_argument("--label-column", default="label")
    text_parser.add_argument("--max-length", type=int, default=128)
    text_parser.add_argument("--train-sample-limit", type=int, default=None)
    text_parser.add_argument("--val-sample-limit", type=int, default=None)

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI service.")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    build_index_parser = subparsers.add_parser(
        "build-index",
        help="Build a small local embedding index from a JSONL file.",
    )
    build_index_parser.add_argument("--records", type=Path, required=True)
    build_index_parser.add_argument("--output", type=Path, required=True)

    search_index_parser = subparsers.add_parser(
        "search-index",
        help="Query a local embedding index with a comma-separated vector.",
    )
    search_index_parser.add_argument("--index-path", type=Path, required=True)
    search_index_parser.add_argument("--vector", type=str, required=True)
    search_index_parser.add_argument("--top-k", type=int, default=5)

    build_text_index_parser = subparsers.add_parser(
        "build-text-index",
        help="Build a transformer text embedding index from JSONL records.",
    )
    build_text_index_parser.add_argument("--records", type=Path, required=True)
    build_text_index_parser.add_argument("--output", type=Path, required=True)
    build_text_index_parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    build_text_index_parser.add_argument("--batch-size", type=int, default=32)
    build_text_index_parser.add_argument("--max-length", type=int, default=256)
    build_text_index_parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)

    search_text_index_parser = subparsers.add_parser(
        "search-text-index",
        help="Query a transformer text embedding index with a natural-language query.",
    )
    search_text_index_parser.add_argument("--index-path", type=Path, required=True)
    search_text_index_parser.add_argument("--query", type=str, required=True)
    search_text_index_parser.add_argument("--top-k", type=int, default=5)
    search_text_index_parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    search_text_index_parser.add_argument("--batch-size", type=int, default=32)
    search_text_index_parser.add_argument("--max-length", type=int, default=256)
    search_text_index_parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)

    al_parser = subparsers.add_parser(
        "active-learning-report",
        help="Generate a relabel queue from model prediction outputs.",
    )
    al_parser.add_argument("--predictions", type=Path, required=True)
    al_parser.add_argument("--output", type=Path, required=True)
    al_parser.add_argument("--limit", type=int, default=100)

    worker_parser = subparsers.add_parser(
        "start-temporal-worker",
        help="Start the Temporal worker for training workflows.",
    )
    worker_parser.add_argument("--task-queue", type=str, default=None)

    check_parser = subparsers.add_parser(
        "check",
        help="Check training environment prerequisites.",
    )
    check_parser.add_argument("--target", choices=("vision", "audio", "text"), default="vision")
    check_parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    check_parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory for artifacts"
    )
    check_parser.add_argument(
        "--dataset-root", type=Path, default=None, help="Dataset root directory"
    )
    check_parser.add_argument("--verbose", action="store_true", help="Show details for all checks")
    check_parser.add_argument("--json", action="store_true", help="Output as JSON")

    return parser


def _parse_vector(raw_vector: str) -> list[float]:
    try:
        return [float(part.strip()) for part in raw_vector.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid vector {raw_vector!r}: {exc}") from exc


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.command == "train-vision":
        from industry_ml_lab.training.vision import train

        config = VisionTrainConfig(
            dataset_root=args.dataset_root or Path("data/vision"),
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size or 64,
            learning_rate=args.learning_rate or 3e-4,
            pretrained=args.pretrained,
            device=args.device,
        )
        metrics = train(config)
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "train-audio":
        from industry_ml_lab.training.audio import train

        config = AudioTrainConfig(
            dataset_root=args.dataset_root or Path("data/audio"),
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size or 128,
            learning_rate=args.learning_rate or 5e-4,
            device=args.device,
        )
        metrics = train(config)
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "train-text-classifier":
        from industry_ml_lab.training.text import train

        config = TextTrainConfig(
            dataset_root=args.dataset_root or Path("data/text"),
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size or 16,
            learning_rate=args.learning_rate or 2e-5,
            device=args.device,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config or None,
            text_column=args.text_column,
            label_column=args.label_column,
            max_length=args.max_length,
            train_sample_limit=args.train_sample_limit,
            val_sample_limit=args.val_sample_limit,
        )
        metrics = train(config)
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "serve":
        import uvicorn

        uvicorn.run("industry_ml_lab.serving.api:app", host=args.host, port=args.port, reload=False)
        return

    if args.command == "build-index":
        from industry_ml_lab.retrieval.simple_index import build_index

        build_index(args.records, args.output)
        print(args.output)
        return

    if args.command == "search-index":
        from industry_ml_lab.retrieval.simple_index import load_index, search_index

        index = load_index(args.index_path)
        results = search_index(index, _parse_vector(args.vector), top_k=args.top_k)
        print(json.dumps(results, indent=2))
        return

    if args.command == "build-text-index":
        from industry_ml_lab.retrieval.text_embeddings import build_text_index

        build_text_index(
            args.records,
            args.output,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
            max_length=args.max_length,
        )
        print(args.output)
        return

    if args.command == "search-text-index":
        from industry_ml_lab.retrieval.text_embeddings import search_text_index

        results = search_text_index(
            args.index_path,
            args.query,
            top_k=args.top_k,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
            max_length=args.max_length,
        )
        print(json.dumps(results, indent=2))
        return

    if args.command == "active-learning-report":
        from industry_ml_lab.active_learning.scoring import write_relabel_queue

        write_relabel_queue(args.predictions, args.output, limit=args.limit)
        print(args.output)
        return

    if args.command == "start-temporal-worker":
        from industry_ml_lab.workflows.worker import run_worker

        run_worker(task_queue=args.task_queue)
        return

    if args.command == "check":
        from industry_ml_lab.training.checklist import run_training_checklist, format_checklist

        checklist = run_training_checklist(
            target=args.target,
            device=args.device,
            output_dir=args.output_dir,
            dataset_root=args.dataset_root,
        )

        if args.json:
            output = json.dumps(
                {
                    "target": checklist.target,
                    "resolved_device": checklist.resolved_device,
                    "is_ready": checklist.is_ready,
                    "has_warnings": checklist.has_warnings,
                    "checks": [
                        {
                            "name": c.name,
                            "status": c.status,
                            "message": c.message,
                            "details": c.details,
                        }
                        for c in checklist.checks
                    ],
                },
                indent=2,
            )
            print(output)
        else:
            print(format_checklist(checklist, verbose=args.verbose))

        if not checklist.is_ready:
            raise SystemExit(1)
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main(sys.argv[1:])
