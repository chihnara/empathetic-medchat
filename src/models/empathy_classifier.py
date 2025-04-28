"""
Empathy classification for medical dialogue.
"""

from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


class EmpathyClassifier:
    """Classifies empathy levels in responses."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: str = "cpu",
    ):
        """Initialize the empathy classifier."""
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,  # low, medium, high
        ).to(device)
        
        # Empathy level mapping
        self.id2label = {0: "low", 1: "medium", 2: "high"}
        
        # Empathy indicators
        self.empathy_indicators = {
            "high": [
                "I understand how challenging",
                "I hear the difficulty",
                "I can see how",
                "I want you to know",
                "Let's work together",
                "This must be very difficult",
                "I'm here to help",
                "completely natural to feel",
                "very concerning",
                "very difficult",
                "incredibly frustrating",
                "discouraging",
                "overwhelming",
                "challenging"
            ],
            "medium": [
                "I understand that",
                "Thank you for sharing",
                "I'd like to understand",
                "Let's discuss",
                "Would you like to talk",
                "concerning",
                "affecting you",
                "manage this",
                "help you",
                "improve your"
            ],
            "low": [
                "I see you're",
                "Regarding your",
                "Could you tell me",
                "What would you like",
                "Let's discuss this",
                "symptoms",
                "conditions",
                "treatment",
                "medication"
            ]
        }

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict empathy level and confidence."""
        try:
            # First check for explicit empathy indicators
            text_lower = text.lower()
            for level, indicators in self.empathy_indicators.items():
                if any(indicator.lower() in text_lower for indicator in indicators):
                    return level, 0.8  # High confidence for explicit indicators
            
            # If no explicit indicators, use the model
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
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            return self.id2label[predicted_class], confidence
            
        except Exception as e:
            print(f"Error in empathy classification: {str(e)}")
            return "medium", 0.5  # Default to medium with low confidence
