PYTHON := uv run

.PHONY: sync lint format train-vision train-audio serve build-index search-index active-learning-report

sync:
	uv sync --extra dev

lint:
	$(PYTHON) ruff check .

format:
	$(PYTHON) ruff format .

train-vision:
	$(PYTHON) ml-lab train-vision --epochs 1 --output-dir artifacts/vision-baseline

train-audio:
	$(PYTHON) ml-lab train-audio --epochs 1 --output-dir artifacts/audio-baseline

serve:
	$(PYTHON) uvicorn industry_ml_lab.serving.api:app --host 0.0.0.0 --port 8000 --reload

build-index:
	$(PYTHON) ml-lab build-index --records sample-data/embedding-records.jsonl --output artifacts/demo-index.json

search-index:
	$(PYTHON) ml-lab search-index --index-path artifacts/demo-index.json --vector 0.92,0.08,0.04

active-learning-report:
	$(PYTHON) ml-lab active-learning-report --predictions sample-data/predictions.jsonl --output artifacts/relabel-queue.csv

