from __future__ import annotations

from pathlib import Path
import time

from industry_ml_lab.config import TextTrainConfig
from industry_ml_lab.training.common import select_device, write_json


def _load_dataset(config: TextTrainConfig):
    from datasets import load_dataset

    kwargs: dict[str, object] = {"cache_dir": str(config.dataset_root)}
    if config.dataset_config:
        return load_dataset(config.dataset_name, config.dataset_config, **kwargs)
    return load_dataset(config.dataset_name, **kwargs)


def _split_datasets(raw_dataset):
    if "train" not in raw_dataset:
        raise ValueError("Dataset must provide a train split")

    train_dataset = raw_dataset["train"]
    if "validation" in raw_dataset:
        val_dataset = raw_dataset["validation"]
    elif "test" in raw_dataset:
        val_dataset = raw_dataset["test"]
    else:
        split = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        val_dataset = split["test"]
    return train_dataset, val_dataset


def _label_names(train_dataset, val_dataset, label_column: str) -> list[str]:
    label_feature = train_dataset.features.get(label_column)
    names = getattr(label_feature, "names", None)
    if names:
        return list(names)

    labels = set(train_dataset[label_column])
    labels.update(val_dataset[label_column])
    return [str(label) for label in sorted(labels)]


def _limit_dataset(dataset, limit: int | None):
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def _tokenize_dataset(dataset, tokenizer, text_column: str, label_column: str, max_length: int):
    keep_columns = {text_column, label_column}
    remove_columns = [column for column in dataset.column_names if column not in keep_columns]

    def tokenize(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=remove_columns + [text_column])
    if label_column != "labels":
        tokenized = tokenized.rename_column(label_column, "labels")
    return tokenized


def _run_epoch(model, loader, optimizer, scheduler, device: str) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        logits = outputs.logits
        labels = batch["labels"]
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


def _evaluate(model, loader, device: str) -> tuple[float, float]:
    import torch

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            labels = batch["labels"]

            total_loss += outputs.loss.item() * labels.size(0)
            total_correct += (outputs.logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


def train(config: TextTrainConfig) -> dict[str, object]:
    from industry_ml_lab.training.checklist import assert_training_ready

    assert_training_ready(
        target="text",
        device=config.device,
        output_dir=config.output_dir,
        dataset_root=config.dataset_root,
    )

    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        get_linear_schedule_with_warmup,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.dataset_root.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)

    raw_dataset = _load_dataset(config)
    train_dataset, val_dataset = _split_datasets(raw_dataset)
    train_dataset = _limit_dataset(train_dataset, config.train_sample_limit)
    val_dataset = _limit_dataset(val_dataset, config.val_sample_limit)

    labels = _label_names(train_dataset, val_dataset, config.label_column)
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_dataset = _tokenize_dataset(
        train_dataset,
        tokenizer,
        config.text_column,
        config.label_column,
        config.max_length,
    )
    val_dataset = _tokenize_dataset(
        val_dataset,
        tokenizer,
        config.text_column,
        config.label_column,
        config.max_length,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = max(1, config.epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    history: list[dict[str, float]] = []
    started_at = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc = _evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

    model_dir = config.output_dir / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    summary = {
        "task": "text-classification",
        "dataset": config.dataset_name,
        "dataset_config": config.dataset_config,
        "model_name": config.model_name,
        "device": device,
        "epochs": config.epochs,
        "train_sample_limit": config.train_sample_limit,
        "val_sample_limit": config.val_sample_limit,
        "duration_seconds": round(time.time() - started_at, 2),
        "best_val_accuracy": max(item["val_accuracy"] for item in history),
        "history": history,
        "artifact_path": str(model_dir),
    }
    write_json(config.output_dir / "metrics.json", summary)
    write_json(
        config.output_dir / "model_meta.json",
        {
            "model_name": config.model_name,
            "architecture": "transformer-sequence-classifier",
            "task": "text-classification",
            "dataset": config.dataset_name,
            "dataset_config": config.dataset_config,
            "labels": labels,
            "text_column": config.text_column,
            "label_column": config.label_column,
            "max_length": config.max_length,
            "artifact_path": str(model_dir),
        },
    )
    return summary
