from __future__ import annotations

from pathlib import Path
import time

from industry_ml_lab.config import VisionTrainConfig
from industry_ml_lab.training.common import select_device, write_json


CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _build_model(num_classes: int, pretrained: bool):
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _run_epoch(model, loader, criterion, optimizer, device: str) -> tuple[float, float]:
    import torch

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += inputs.size(0)

    return total_loss / total_examples, total_correct / total_examples


def _evaluate(model, loader, criterion, device: str) -> tuple[float, float]:
    import torch

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += inputs.size(0)

    return total_loss / total_examples, total_correct / total_examples


def train(config: VisionTrainConfig) -> dict[str, object]:
    from industry_ml_lab.training.checklist import assert_training_ready

    assert_training_ready(
        target="vision",
        device=config.device,
        output_dir=config.output_dir,
        dataset_root=config.dataset_root,
    )

    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.dataset_root.mkdir(parents=True, exist_ok=True)

    device = select_device(config.device)

    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    eval_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(config.dataset_root, train=True, download=True, transform=train_transforms)
    val_dataset = datasets.CIFAR10(config.dataset_root, train=False, download=True, transform=eval_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device == "cuda",
    )

    model = _build_model(num_classes=len(CIFAR10_LABELS), pretrained=config.pretrained).to(device)
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

    model_path = config.output_dir / "model.pt"
    torch.save(
        {
            "architecture": "resnet18",
            "labels": CIFAR10_LABELS,
            "image_size": 224,
            "state_dict": model.state_dict(),
        },
        model_path,
    )

    summary = {
        "task": "vision-classification",
        "dataset": "CIFAR10",
        "device": device,
        "epochs": config.epochs,
        "pretrained": config.pretrained,
        "duration_seconds": round(time.time() - started_at, 2),
        "best_val_accuracy": max(item["val_accuracy"] for item in history),
        "history": history,
        "artifact_path": str(model_path),
    }
    write_json(config.output_dir / "metrics.json", summary)
    write_json(
        config.output_dir / "model_meta.json",
        {
            "model_name": "vision-baseline",
            "architecture": "resnet18",
            "labels": CIFAR10_LABELS,
            "image_size": 224,
            "dataset": "CIFAR10",
            "artifact_path": str(model_path),
        },
    )
    return summary
