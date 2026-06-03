"""CLI command for training Marcus Aurelius philosopher model."""

from __future__ import annotations

import typer

from industry_ml_lab.training.philosopher import train
from industry_ml_lab.training.philosopher_config import PhilosopherTrainConfig, load_config


def main(
    config_path: str | None = typer.Option(None, help="Path to config JSON file"),
    model_name: str = typer.Option("unsloth/Qwen3-100M", help="Base model to train"),
    dataset_path: str = typer.Option("marcus_training_data.json", help="Training data JSON path"),
    output_dir: str = typer.Option("artifacts/philosopher", help="Output directory"),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    batch_size: int = typer.Option(4, help="Batch size"),
    learning_rate: float = typer.Option(2e-4, help="Learning rate"),
    max_length: int = typer.Option(2048, help="Maximum sequence length"),
    device: str = typer.Option("cuda", help="Device: auto, cuda, mps, cpu"),
):
    """Train a tiny LLM to be a Marcus Aurelius philosopher."""
    config = PhilosopherTrainConfig(
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        device=device,
    )
    
    if config_path:
        config = load_config(config_path)
    
    print(f"Training philosopher model: {config.model_name}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Output: {config.output_dir}")
    
    results = train(config)
    
    print(f"\nTraining complete!")
    print(f"Final train loss: {results['final_train_loss']:.4f}")
    if results['final_val_loss']:
        print(f"Final val loss: {results['final_val_loss']:.4f}")


if __name__ == "__main__":
    typer.run(main)
