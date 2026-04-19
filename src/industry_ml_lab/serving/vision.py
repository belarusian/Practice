from __future__ import annotations

from io import BytesIO
from pathlib import Path


class VisionPredictor:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._model = None
        self._labels: list[str] | None = None

    def _load(self) -> None:
        import torch
        from torch import nn
        from torchvision import transforms
        from torchvision.models import resnet18

        checkpoint = torch.load(self.model_path, map_location="cpu")
        labels = checkpoint["labels"]
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(labels))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        self._model = model
        self._labels = labels
        self._transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def predict_bytes(self, image_bytes: bytes, top_k: int = 3) -> list[dict[str, float | str]]:
        from PIL import Image
        import torch

        if self._model is None or self._labels is None:
            self._load()

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = self._transform(image).unsqueeze(0)

        with torch.inference_mode():
            logits = self._model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        values, indices = torch.topk(probabilities, k=min(top_k, len(self._labels)))
        return [
            {
                "label": self._labels[index.item()],
                "score": round(value.item(), 6),
            }
            for value, index in zip(values, indices, strict=True)
        ]

