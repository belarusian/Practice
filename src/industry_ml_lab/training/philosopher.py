"""Training module for Marcus Aurelius philosopher model."""

from __future__ import annotations

from pathlib import Path
import time

from industry_ml_lab.config import PhilosopherTrainConfig
from industry_ml_lab.training.common import select_device, write_json


def _load_dataset(config: PhilosopherTrainConfig):
    """Load training data from JSON file."""
    import json
    
    with open(config.dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to Hugging Face dataset format
    from datasets import Dataset
    
    formatted_data = []
    for item in data:
        messages = item.get("messages", [])
        # Format as chat template
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text += f"<|{role}|>{content}<|end|>\n"
        
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)


def _split_datasets(raw_dataset):
    """Split dataset into train and validation."""
    if "train" not in raw_dataset.column_names:
        split = raw_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        val_dataset = split["test"]
    else:
        train_dataset = raw_dataset
        val_dataset = None
    return train_dataset, val_dataset


def _limit_dataset(dataset, limit: int | None):
    """Limit dataset size."""
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def _tokenize_dataset(dataset, tokenizer, max_length: int):
    """Tokenize dataset."""
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    
    tokenized = dataset.map(tokenize, batched=True)
    return tokenized


def train(config: PhilosopherTrainConfig) -> dict[str, object]:
    """Train a tiny philosopher LLM."""
    from industry_ml_lab.training.checklist import assert_training_ready
    
    assert_training_ready(
        target="philosopher",
        device=config.device,
        output_dir=config.output_dir,
        dataset_root=Path(config.dataset_path).parent,
    )
    
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        DataCollatorForLanguageModeling,
        get_linear_schedule_with_warmup,
    )
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)
    
    # Load dataset
    raw_dataset = _load_dataset(config)
    train_dataset, val_dataset = _split_datasets(raw_dataset)
    train_dataset = _limit_dataset(train_dataset, config.train_sample_limit)
    val_dataset = _limit_dataset(val_dataset, config.val_sample_limit)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        num_labels=1,
        torch_dtype=config.torch_dtype,
    ).to(device)
    
    # Tokenize datasets
    train_dataset = _tokenize_dataset(train_dataset, tokenizer, config.max_length)
    if val_dataset:
        val_dataset = _tokenize_dataset(val_dataset, tokenizer, config.max_length)
    
    # Data collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collator,
        )
    else:
        val_loader = None
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = max(1, config.epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    
    # Training loop
    history: list[dict[str, float]] = []
    started_at = time.time()
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item() * batch["input_ids"].size(0)
            total_examples += batch["input_ids"].size(0)
        
        train_loss = total_loss / total_examples
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_examples = 0
            
            with torch.inference_mode():
                for batch in val_loader:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    outputs = model(**batch)
                    val_loss += outputs.loss.item() * batch["input_ids"].size(0)
                    val_examples += batch["input_ids"].size(0)
            
            val_loss = val_loss / val_examples
        else:
            val_loss = None
        
        history.append({
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed": time.time() - started_at,
        })
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss}")
    
    # Save model
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # Save training history
    write_json(history, config.output_dir / "training_history.json")
    
    return {
        "history": history,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1].get("val_loss"),
    }
