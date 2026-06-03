"""Configuration for Marcus Aurelius philosopher model training."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class PhilosopherTrainConfig(BaseModel):
    """Configuration for philosopher model training."""
    
    # Model configuration
    model_name: str = "unsloth/Qwen3-100M"  # Tiny model for philosophy
    torch_dtype: str = "float16"
    
    # Dataset configuration
    dataset_path: Path = Path("marcus_training_data.json")
    train_sample_limit: int | None = None
    val_sample_limit: int | None = None
    
    # Training configuration
    output_dir: Path = Path("artifacts/philosopher")
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_length: int = 2048
    
    # Hardware configuration
    device: str = "cuda"  # auto, cuda, mps, cpu
    
    # Data loader configuration
    num_workers: int = 0
    
    class Config:
        arbitrary_types_allowed = True


def load_config(path: str | Path | None = None) -> PhilosopherTrainConfig:
    """Load configuration from file or use defaults."""
    if path is None:
        return PhilosopherTrainConfig()
    
    import json
    with open(path, 'r') as f:
        data = json.load(f)
    return PhilosopherTrainConfig(**data)
