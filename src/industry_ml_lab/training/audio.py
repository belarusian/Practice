from __future__ import annotations

import os
from pathlib import Path
import time

from industry_ml_lab.config import AudioTrainConfig
from industry_ml_lab.training.common import select_device, write_json


def _subset_dataset(root: Path, subset: str):
    import torchaudio

    class SpeechCommandsSubset(torchaudio.datasets.SPEECHCOMMANDS):
        def __init__(self, dataset_root: str, subset_name: str):
            super().__init__(dataset_root, download=True)

            def _load_list(filename: str) -> list[str]:
                list_path = os.path.join(self._path, filename)
                with open(list_path, encoding="utf-8") as handle:
                    return [os.path.normpath(os.path.join(self._path, line.strip())) for line in handle]

            if subset_name == "validation":
                self._walker = _load_list("validation_list.txt")
            elif subset_name == "testing":
                self._walker = _load_list("testing_list.txt")
            elif subset_name == "training":
                excluded = set(_load_list("validation_list.txt") + _load_list("testing_list.txt"))
                self._walker = [path for path in self._walker if path not in excluded]

    return SpeechCommandsSubset(str(root), subset)


def _prepare_label_maps(dataset) -> tuple[list[str], dict[str, int]]:
    labels = sorted({sample[2] for sample in dataset})
    return labels, {label: index for index, label in enumerate(labels)}


def _collate_batch(batch, label_to_index):
    import torch
    import torchaudio.transforms as T
    from torch.nn import functional as F

    mel_transform = T.MelSpectrogram(sample_rate=16_000, n_mels=64)
    db_transform = T.AmplitudeToDB()

    features = []
    targets = []
    for waveform, sample_rate, label, *_ in batch:
        if sample_rate != 16_000:
            raise ValueError(f"Expected 16kHz audio, got {sample_rate}")
        mono = waveform.mean(dim=0, keepdim=True)
        if mono.size(1) < 16_000:
            mono = F.pad(mono, (0, 16_000 - mono.size(1)))
        else:
            mono = mono[:, :16_000]
        mel = db_transform(mel_transform(mono))
        features.append(mel)
        targets.append(label_to_index[label])

    return torch.stack(features), torch.tensor(targets)


def _build_model(num_classes: int):
    from torch import nn

    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, num_classes),
    )


def _run_epoch(model, loader, criterion, optimizer, device: str) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += features.size(0)

    return total_loss / total_examples, total_correct / total_examples


def _evaluate(model, loader, criterion, device: str) -> tuple[float, float]:
    import torch

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item() * features.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += features.size(0)

    return total_loss / total_examples, total_correct / total_examples


def train(config: AudioTrainConfig) -> dict[str, object]:
    from torch import nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.dataset_root.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)

    train_dataset = _subset_dataset(config.dataset_root, "training")
    val_dataset = _subset_dataset(config.dataset_root, "validation")
    labels, label_to_index = _prepare_label_maps(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda batch: _collate_batch(batch, label_to_index),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=lambda batch: _collate_batch(batch, label_to_index),
    )

    model = _build_model(len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    history: list[dict[str, float]] = []
    started_at = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

    import torch

    model_path = config.output_dir / "model.pt"
    torch.save(
        {
            "architecture": "speechcommands-cnn",
            "labels": labels,
            "state_dict": model.state_dict(),
        },
        model_path,
    )
    summary = {
        "task": "audio-classification",
        "dataset": "SpeechCommands",
        "device": device,
        "epochs": config.epochs,
        "duration_seconds": round(time.time() - started_at, 2),
        "best_val_accuracy": max(item["val_accuracy"] for item in history),
        "history": history,
        "artifact_path": str(model_path),
    }
    write_json(config.output_dir / "metrics.json", summary)
    write_json(
        config.output_dir / "model_meta.json",
        {
            "model_name": "audio-baseline",
            "architecture": "speechcommands-cnn",
            "labels": labels,
            "dataset": "SpeechCommands",
            "artifact_path": str(model_path),
        },
    )
    return summary

