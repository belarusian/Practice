from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class PredictionLabel(BaseModel):
    label: str
    score: float


class VisionPredictionResponse(BaseModel):
    model_name: str
    predictions: list[PredictionLabel]


class EmbeddingSearchRequest(BaseModel):
    vector: list[float]
    top_k: int = Field(default=5, ge=1, le=50)


class EmbeddingMatch(BaseModel):
    item_id: str
    score: float
    metadata: dict[str, Any]


class EmbeddingSearchResponse(BaseModel):
    matches: list[EmbeddingMatch]

