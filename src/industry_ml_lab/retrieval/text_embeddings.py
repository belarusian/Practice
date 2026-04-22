from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

from industry_ml_lab.retrieval.simple_index import _vector_norm, load_index, search_index


DEFAULT_TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(slots=True)
class TextRecord:
    item_id: str
    text: str
    metadata: dict[str, Any]


class TextEmbedder(Protocol):
    model_name: str

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


class TransformerTextEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_EMBEDDING_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 256,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Transformer text embeddings require the transformer extra. "
                "Install with: uv sync --extra transformer"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._torch = torch
        self.device = device or _resolve_device(torch)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        torch = self._torch
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"], torch)
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(
                    [[float(value) for value in row] for row in normalized.cpu().tolist()]
                )
        return vectors


def _resolve_device(torch: Any) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _mean_pool(last_hidden_state: Any, attention_mask: Any, torch: Any) -> Any:
    expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * expanded_mask, dim=1)
    counts = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
    return summed / counts


def _read_text_records(path: Path) -> list[TextRecord]:
    records: list[TextRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if "id" not in raw or "text" not in raw:
                raise ValueError(f"{path}:{line_number} must contain 'id' and 'text'")
            metadata = raw.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"{path}:{line_number} metadata must be an object")
            text = str(raw["text"]).strip()
            if not text:
                raise ValueError(f"{path}:{line_number} text must not be empty")
            records.append(TextRecord(item_id=str(raw["id"]), text=text, metadata=metadata))
    if not records:
        raise ValueError(f"No text records found in {path}")
    return records


def build_text_index(
    records_path: Path,
    output_path: Path,
    *,
    embedder: TextEmbedder | None = None,
    model_name: str = DEFAULT_TEXT_EMBEDDING_MODEL,
    batch_size: int = 32,
    device: str | None = None,
    max_length: int = 256,
) -> None:
    records = _read_text_records(records_path)
    embedder = embedder or TransformerTextEmbedder(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )
    vectors = embedder.embed_texts([record.text for record in records])
    if len(vectors) != len(records):
        raise ValueError("Embedder returned a different number of vectors than input records.")
    if not vectors or not vectors[0]:
        raise ValueError("Embedder returned empty vectors.")

    width = len(vectors[0])
    indexed = []
    for record, vector_values in zip(records, vectors, strict=True):
        vector = [float(value) for value in vector_values]
        if len(vector) != width:
            raise ValueError("Embedding dimensions are inconsistent.")
        metadata = dict(record.metadata)
        metadata.setdefault("text", record.text)
        indexed.append(
            {
                "item_id": record.item_id,
                "vector": vector,
                "norm": _vector_norm(vector),
                "metadata": metadata,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "dimension": width,
                "embedding_model": getattr(embedder, "model_name", model_name),
                "source": str(records_path),
                "records": indexed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def search_text_index(
    index_path: Path,
    query: str,
    *,
    embedder: TextEmbedder | None = None,
    top_k: int = 5,
    model_name: str = DEFAULT_TEXT_EMBEDDING_MODEL,
    batch_size: int = 32,
    device: str | None = None,
    max_length: int = 256,
) -> list[dict[str, Any]]:
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")

    embedder = embedder or TransformerTextEmbedder(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )
    query_vectors = embedder.embed_texts([query])
    if len(query_vectors) != 1 or not query_vectors[0]:
        raise ValueError("Embedder did not return exactly one query vector.")
    return search_index(load_index(index_path), query_vectors[0], top_k=top_k)
