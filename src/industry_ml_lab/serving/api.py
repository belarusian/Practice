from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile

from industry_ml_lab.config import AppSettings
from industry_ml_lab.retrieval.simple_index import load_index, search_index
from industry_ml_lab.serving.schemas import (
    EmbeddingSearchResponse,
    EmbeddingSearchRequest,
    HealthResponse,
    VisionPredictionResponse,
)
from industry_ml_lab.serving.vision import VisionPredictor


app = FastAPI(title="Industry ML Lab API", version="0.1.0")


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings.from_env()


@lru_cache
def get_predictor() -> VisionPredictor:
    settings = get_settings()
    if not settings.model_path.exists():
        raise FileNotFoundError(settings.model_path)
    return VisionPredictor(settings.model_path)


@lru_cache
def get_embedding_index():
    settings = get_settings()
    if not settings.index_path.exists():
        raise FileNotFoundError(settings.index_path)
    return load_index(settings.index_path)


@app.get("/healthz", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", service="industry-ml-lab-api", version=settings.service_version)


@app.get("/readyz", response_model=HealthResponse)
def readiness() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ready", service="industry-ml-lab-api", version=settings.service_version)


@app.post("/vision/predict", response_model=VisionPredictionResponse)
async def predict_vision(file: UploadFile = File(...), top_k: int = 3) -> VisionPredictionResponse:
    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model artifact not found: {exc}") from exc

    payload = await file.read()
    predictions = predictor.predict_bytes(payload, top_k=top_k)
    return VisionPredictionResponse(model_name="vision-baseline", predictions=predictions)


@app.post("/embeddings/search", response_model=EmbeddingSearchResponse)
def embedding_search(request: EmbeddingSearchRequest) -> EmbeddingSearchResponse:
    try:
        index = get_embedding_index()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Embedding index not found: {exc}") from exc

    matches = search_index(index, request.vector, top_k=request.top_k)
    return EmbeddingSearchResponse(matches=matches)

