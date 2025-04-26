"""
Training script for medical NER model.
"""

import os
import torch
from typing import Dict, List, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import classification_report
import json


class MedicalNERDataset:
    """Dataset for medical NER training."""

    # Label mapping as class attribute
    label2id = {
        "O": 0,
        "B-SYMPTOM": 1,
        "I-SYMPTOM": 2,
        "B-CONDITION": 3,
        "I-CONDITION": 4,
        "B-TREATMENT": 5,
        "I-TREATMENT": 6,
        "B-MEDICATION": 7,
        "I-MEDICATION": 8,
    }
    id2label = {v: k for k, v in label2id.items()}

    def __init__(
        self,
        texts: List[str],
        labels: List[List[str]],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = self._create_dataset()

    def _create_dataset(self) -> Dataset:
        """Create HuggingFace dataset from texts and labels."""
        # Tokenize texts
        tokenized_inputs = self.tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            is_split_into_words=True,
            return_tensors="pt",
        )

        # Align labels with tokens
        label_ids = []
        for i, label in enumerate(self.labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_id = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_id.append(-100)
                elif word_idx != previous_word_idx:
                    try:
                        label_id.append(self.label2id[label[word_idx]])
                    except IndexError:
                        label_id.append(self.label2id["O"])
                else:
                    try:
                        label_id.append(self.label2id[label[word_idx]])
                    except IndexError:
                        label_id.append(self.label2id["O"])
                previous_word_idx = word_idx

            label_ids.append(label_id)

        tokenized_inputs["labels"] = label_ids
        return Dataset.from_dict(tokenized_inputs)


def compute_metrics(eval_preds):
    """Compute metrics for NER evaluation."""
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Flatten predictions and labels
    true_predictions = [p for pred in true_predictions for p in pred]
    true_labels = [l for label in true_labels for l in label]

    # Get unique labels
    unique_labels = sorted(set(true_labels))
    label_names = [MedicalNERDataset.id2label[i] for i in unique_labels]

    # Compute metrics
    report = classification_report(
        true_labels,
        true_predictions,
        labels=unique_labels,
        target_names=label_names,
        output_dict=True,
    )

    return {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "accuracy": report["accuracy"],
    }


def train_medical_ner(
    train_data: List[Dict[str, List[str]]],
    val_data: List[Dict[str, List[str]]],
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "models/medical_ner",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
):
    """Train medical NER model."""
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(MedicalNERDataset.label2id),
        id2label=MedicalNERDataset.id2label,
        label2id=MedicalNERDataset.label2id,
    )

    # Create datasets
    train_texts = [item["text"].split() for item in train_data]
    train_labels = [item["labels"] for item in train_data]
    val_texts = [item["text"].split() for item in val_data]
    val_labels = [item["labels"] for item in val_data]

    train_dataset = MedicalNERDataset(
        train_texts, train_labels, tokenizer, max_length
    ).dataset
    val_dataset = MedicalNERDataset(
        val_texts, val_labels, tokenizer, max_length
    ).dataset

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()

    # Save model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


if __name__ == "__main__":
    # Example usage
    train_data = [
        {
            "text": "I have a severe headache and dizziness",
            "labels": ["O", "O", "O", "B-SYMPTOM", "I-SYMPTOM", "O", "B-SYMPTOM"],
        },
        {
            "text": "My chronic back pain is getting worse",
            "labels": ["O", "B-CONDITION", "B-SYMPTOM", "I-SYMPTOM", "O", "O", "O"],
        },
    ]

    val_data = [
        {
            "text": "I've been diagnosed with high blood pressure",
            "labels": ["O", "O", "O", "O", "B-CONDITION", "I-CONDITION", "I-CONDITION"],
        },
    ]

    model, tokenizer = train_medical_ner(
        train_data=train_data,
        val_data=val_data,
        output_dir="models/medical_ner",
    )
