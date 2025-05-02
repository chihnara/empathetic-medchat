"""
Empathy classification for medical dialogue (v1).
"""

from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EmpathyClassifierV1:
    """Classifies empathy levels in medical dialogue."""

    def __init__(
        self, model_name: str = "distilbert-base-uncased", device: str = "cpu"
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,  # low, medium, high
        ).to(device)

        # Empathy level mapping
        self.empathy_levels = {0: "low", 1: "medium", 2: "high"}

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict empathy level and confidence."""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)

                # Get prediction and confidence
                pred_idx = torch.argmax(probabilities).item()
                confidence = probabilities[0][pred_idx].item()

                return self.empathy_levels[pred_idx], confidence

        except Exception as e:
            print(f"Error in empathy classification: {str(e)}")
            return "medium", 0.5  # Default values
