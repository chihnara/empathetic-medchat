"""
Enhanced empathy classification for medical dialogue with fine-grained emotional states.
"""

from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


class EnhancedEmpathyClassifier:
    """Enhanced empathy classifier with fine-grained emotional states."""

    def __init__(
        self, model_name: str = "distilbert-base-uncased", device: str = "cpu"
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=8,  # Extended emotional states
        ).to(device)

        # Extended emotion categories
        self.emotion_categories = {
            "distress": 0,
            "anxiety": 1,
            "concern": 2,
            "discomfort": 3,
            "frustration": 4,
            "relief": 5,
            "hopeful": 6,
            "neutral": 7,
        }

        # Multi-level classification
        self.empathy_levels = {
            "high": ["distress", "anxiety", "concern"],
            "medium": ["discomfort", "frustration", "relief"],
            "low": ["hopeful", "neutral"],
        }

        # Enhanced emotion indicators
        self.emotion_indicators = {
            "distress": [
                "severe",
                "unbearable",
                "terrible",
                "extreme",
                "intense pain",
                "very bad",
                "emergency",
            ],
            "anxiety": [
                "worried",
                "anxious",
                "scared",
                "nervous",
                "fear",
                "panic",
                "stress",
            ],
            "concern": [
                "concerned",
                "troubling",
                "bothering",
                "persistent",
                "getting worse",
                "not improving",
            ],
            "discomfort": [
                "uncomfortable",
                "annoying",
                "irritating",
                "mild pain",
                "ache",
                "sore",
            ],
            "frustration": [
                "frustrated",
                "tired of",
                "fed up",
                "annoyed",
                "irritated",
                "not helping",
            ],
            "relief": [
                "better",
                "improving",
                "relieved",
                "helping",
                "working",
                "progress",
            ],
            "hopeful": [
                "hopeful",
                "optimistic",
                "positive",
                "confident",
                "getting better",
            ],
            "neutral": ["okay", "fine", "normal", "stable", "unchanged"],
        }

    def predict_multilevel(self, text: str) -> Dict:
        """Predict multiple emotional aspects of the response."""
        try:
            emotions = []
            confidences = {}

            # Check explicit indicators
            text_lower = text.lower()
            for emotion, indicators in self.emotion_indicators.items():
                if any(indicator in text_lower for indicator in indicators):
                    emotions.append(emotion)
                    confidences[emotion] = (
                        0.9  # High confidence for explicit indicators
                    )

            # Model prediction if no explicit indicators found
            if not emotions:
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

                    # Get top 2 predictions with confidence > 0.3
                    top_probs, top_indices = torch.topk(probabilities, k=2)
                    for prob, idx in zip(top_probs[0], top_indices[0]):
                        if prob.item() > 0.3:
                            emotion = list(self.emotion_categories.keys())[idx]
                            emotions.append(emotion)
                            confidences[emotion] = prob.item()

            # Default to neutral if no emotions detected
            if not emotions:
                emotions = ["neutral"]
                confidences["neutral"] = 0.5

            # Determine overall empathy level
            empathy_level = "low"
            for level, level_emotions in self.empathy_levels.items():
                if any(emotion in level_emotions for emotion in emotions):
                    empathy_level = level
                    break

            # Calculate overall confidence
            overall_confidence = max(confidences.values()) if confidences else 0.5

            return {
                "emotions": emotions,
                "confidences": confidences,
                "empathy_level": empathy_level,
                "overall_confidence": overall_confidence,
            }

        except Exception as e:
            print(f"Error in emotion classification: {str(e)}")
            return {
                "emotions": ["neutral"],
                "confidences": {"neutral": 0.5},
                "empathy_level": "medium",
                "overall_confidence": 0.5,
            }
