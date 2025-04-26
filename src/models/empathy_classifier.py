"""
Empathy classification for medical dialogue.
"""

from typing import Dict, List, Tuple
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import joblib
from pathlib import Path
from functools import lru_cache


class EmpathyDataset(Dataset):
    """Dataset for empathy classification."""

    def __init__(
        self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = self._preprocess_texts()

    def _preprocess_texts(self):
        """Preprocess all texts at once for better performance."""
        return self.tokenizer(
            self.texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class EmpathyClassifier:
    """Classifies empathy levels in medical responses."""

    def __init__(
        self,
        model_name: str = "bhadresh-savani/distilbert-base-uncased-emotion",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the empathy classifier."""
        self.device = device
        self.model_name = model_name

        # Initialize classifier pipeline
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=-1,  # Force CPU
        )

        # Emotion to empathy level mapping
        self.emotion_to_empathy = {
            "joy": "high",
            "sadness": "medium",
            "anger": "low",
            "fear": "medium",
            "love": "high",
            "surprise": "medium",
        }

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict empathy level and confidence score."""
        try:
            # Get predictions
            prediction = self.classifier(text)[0]

            if prediction and isinstance(prediction, dict):
                emotion = prediction.get("label", "").lower()
                confidence = float(prediction.get("score", 0.0))

                # Map emotion to empathy level
                empathy_level = self.emotion_to_empathy.get(emotion, "medium")

                return empathy_level, confidence

            return "medium", 0.5

        except Exception as e:
            print(f"Error in empathy classification: {str(e)}")
            return "medium", 0.5

    def _map_to_empathy_level(self, emotion: str) -> str:
        """Map emotion to empathy level."""
        return self.emotion_to_empathy.get(emotion.lower(), "medium")

    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        output_dir: str = "./empathy_model",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ):
        """Train the empathy classifier."""
        # Create datasets with preprocessed texts
        train_dataset = EmpathyDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = EmpathyDataset(val_texts, val_labels, self.tokenizer)

        # Define optimized training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            fp16=True,  # Enable mixed precision training
            gradient_accumulation_steps=4,  # Accumulate gradients
            dataloader_num_workers=4,  # Parallel data loading
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
        )

        # Train model
        trainer.train()

        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict empathy levels for a batch of texts."""
        predictions = []
        for text in texts:
            predictions.append(self.predict(text))
        return predictions

    def _compute_metrics(self, pred):
        """Compute metrics for evaluation."""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted"
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
