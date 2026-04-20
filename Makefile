PYTHON := uv run

.PHONY: sync sync-lite lint format test check training-mode sunny-proof sunny-proof-audio sunny-vision-smoke sunny-audio-smoke train-vision train-audio serve build-index search-index active-learning-report aws-bootstrap aws-budget aws-upload-artifacts aws-launch-trainer aws-build-and-push-api aws-launch-api

sync:
	uv sync --extra dev --extra serving --extra training --extra vector --extra workflow

sync-lite:
	uv sync --extra dev

lint:
	$(PYTHON) ruff check .

format:
	$(PYTHON) ruff format .

test:
	$(PYTHON) pytest

check:
	$(PYTHON) ml-lab check --verbose

training-mode:
	$(PYTHON) ml-lab check --target vision --verbose --json | python -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d['is_ready'] else 1)"

sunny-proof:
	bash ops/sunny/run-remote-training-proof.sh --with-training-mode --restore-demo

sunny-proof-audio:
	bash ops/sunny/run-remote-training-proof.sh --with-training-mode --restore-demo --target audio

sunny-vision-smoke:
	bash ops/sunny/run-remote-vision-smoke.sh

sunny-audio-smoke:
	bash ops/sunny/run-remote-audio-smoke.sh

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

aws-bootstrap:
	./infra/aws/scripts/bootstrap.sh

aws-budget:
	./infra/aws/scripts/create-budget.sh

aws-upload-artifacts:
	./infra/aws/scripts/upload-artifacts.sh

aws-launch-trainer:
	./infra/aws/scripts/launch-spot-trainer.sh

aws-build-and-push-api:
	./infra/aws/scripts/build-and-push-api.sh

aws-launch-api:
	./infra/aws/scripts/launch-api-instance.sh
