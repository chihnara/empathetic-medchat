"""
Empathy classification model implementation.
"""

from typing import Dict, Optional, Tuple, Union
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

from ..config.config import MODEL_CONFIG, MODEL_DIR


class EmpathyClassifier:
    """Empathy classification model wrapper."""

    def __init__(
        self,
        model: Optional[
            Union[AutoModelForSequenceClassification, LogisticRegression]
        ] = None,
        tokenizer: Optional[Union[AutoTokenizer, TfidfVectorizer]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def load(cls, save_dir: str = MODEL_DIR / "empathy") -> "EmpathyClassifier":
        """Load model and tokenizer from disk."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Try loading transformer model first
        try:
            model = AutoModelForSequenceClassification.from_pretrained(save_dir)
            tokenizer = AutoTokenizer.from_pretrained(save_dir)
            return cls(model, tokenizer)
        except:
            # Fall back to traditional ML model
            model = joblib.load(save_dir / "model.joblib")
            tokenizer = joblib.load(save_dir / "vectorizer.joblib")
            return cls(model, tokenizer)

    def save(self, save_dir: str = MODEL_DIR / "empathy") -> None:
        """Save model and tokenizer to disk."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if isinstance(self.model, AutoModelForSequenceClassification):
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
        else:
            joblib.dump(self.model, save_dir / "model.joblib")
            joblib.dump(self.tokenizer, save_dir / "vectorizer.joblib")

    def predict(self, text: str) -> Dict:
        """Predict empathy level for given text."""
        if isinstance(self.model, AutoModelForSequenceClassification):
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=MODEL_CONFIG["empathy"]["max_length"],
                truncation=True,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                confidence = torch.max(probabilities).item()
                predicted_idx = torch.argmax(logits).item()

                empathy_levels = ["low", "medium", "high"]
                predicted_level = empathy_levels[predicted_idx]
        else:
            text_features = self.tokenizer.transform([text])
            probabilities = self.model.predict_proba(text_features)[0]
            confidence = max(probabilities)
            predicted_level = self.model.predict(text_features)[0]

            if predicted_level not in ["low", "medium", "high"]:
                predicted_level = "low"

        return {"empathy_level": predicted_level, "confidence": float(confidence)}
