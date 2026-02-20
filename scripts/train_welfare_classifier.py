#!/usr/bin/env python3
"""
Train welfare construct classifier using DistilBERT.

Usage:
    python scripts/train_welfare_classifier.py \
        --train-data data/training/welfare_training_split_train.jsonl \
        --val-data data/training/welfare_training_split_val.jsonl \
        --output-dir models/welfare-constructs-distilbert

Time: 2-3 hours on L8s_v3 (no GPU), 30 minutes on GPU
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONSTRUCT_NAMES = ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]


class WelfareDataset(Dataset):
    """Multi-label welfare construct dataset."""

    def __init__(self, jsonl_path: Path, tokenizer, max_length: int = 512):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(f"Loaded {len(self.data)} examples from {jsonl_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Create label tensor [8] with scores [0,1]
        labels = torch.tensor([
            item['scores'][construct]
            for construct in CONSTRUCT_NAMES
        ], dtype=torch.float)

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels
        }


def compute_metrics(eval_pred):
    """
    Compute regression metrics for multi-label continuous scores.
    """
    predictions, labels = eval_pred

    # Convert logits to probabilities [0,1]
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()

    # Overall metrics
    mae = mean_absolute_error(labels.flatten(), predictions.flatten())
    mse = mean_squared_error(labels.flatten(), predictions.flatten())

    # Per-construct metrics
    per_construct = {}
    for i, name in enumerate(CONSTRUCT_NAMES):
        per_construct[f"mae_{name}"] = mean_absolute_error(
            labels[:, i], predictions[:, i]
        )

    return {
        "mae": mae,
        "mse": mse,
        **per_construct
    }


def main():
    parser = argparse.ArgumentParser(description='Train welfare classifier')
    parser.add_argument('--train-data', type=Path, required=True,
                       help='Training JSONL file')
    parser.add_argument('--val-data', type=Path, required=True,
                       help='Validation JSONL file')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate')

    args = parser.parse_args()

    logger.info("Loading tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=8,  # 8 constructs
        problem_type="multi_label_classification"
    )

    logger.info("Loading datasets...")
    train_dataset = WelfareDataset(args.train_data, tokenizer)
    val_dataset = WelfareDataset(args.val_data, tokenizer)

    logger.info(f"Training on {len(train_dataset)} examples")
    logger.info(f"Validating on {len(val_dataset)} examples")

    # Training configuration
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),  # Use FP16 if GPU available
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info("Starting training...")
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("OOM - reducing batch size and retrying...")
            training_args.per_device_train_batch_size = 8
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
            )
            trainer.train()
        else:
            raise

    # Evaluate
    logger.info("Evaluating final model...")
    metrics = trainer.evaluate()

    logger.info("Final metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    # Save model and tokenizer
    logger.info(f"Saving model to {args.output_dir}...")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    logger.info("✓ Training complete!")

    # Check success criteria
    overall_mae = metrics.get('eval_mae', 1.0)
    if overall_mae < 0.20:
        logger.info(f"✓ SUCCESS: MAE {overall_mae:.4f} < 0.20 (target met)")
    else:
        logger.warning(f"⚠ WARNING: MAE {overall_mae:.4f} >= 0.20 (target not met)")


if __name__ == '__main__':
    main()
