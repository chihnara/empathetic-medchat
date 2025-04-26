"""
Context analysis utilities for medical dialogue.
"""

from typing import Dict, List, Tuple
import re
from ..config.config import EMOTIONAL_CONTEXT, MEDICAL_ADVICE_TEMPLATES


class ContextAnalyzer:
    """Analyzes emotional and medical context in text."""

    @staticmethod
    def extract_emotional_context(text: str) -> Dict[str, float]:
        """Extract emotional context from text."""
        text = text.lower()
        emotions = {}

        for emotion, keywords in EMOTIONAL_CONTEXT["keywords"].items():
            count = sum(1 for keyword in keywords if keyword in text)
            if count > 0:
                emotions[emotion] = count / len(keywords)

        return emotions

    @staticmethod
    def get_emotional_intensity(emotions: Dict[str, float]) -> str:
        """Get emotional intensity level."""
        if not emotions:
            return "low"

        max_intensity = max(emotions.values())
        if max_intensity >= EMOTIONAL_CONTEXT["intensity_thresholds"]["high"]:
            return "high"
        elif max_intensity >= EMOTIONAL_CONTEXT["intensity_thresholds"]["medium"]:
            return "medium"
        return "low"

    @staticmethod
    def extract_medical_context(text: str) -> List[str]:
        """Extract medical symptoms and conditions from text."""
        symptoms = []
        text = text.lower()

        for category, template in MEDICAL_ADVICE_TEMPLATES.items():
            if any(keyword in text for keyword in template["keywords"]):
                symptoms.append(category)

        return symptoms

    @staticmethod
    def get_medical_advice(symptoms: List[str]) -> str:
        """Get medical advice for identified symptoms."""
        if not symptoms:
            return "Please monitor your symptoms and maintain a healthy lifestyle."

        advice = []
        for symptom in symptoms:
            if symptom in MEDICAL_ADVICE_TEMPLATES:
                advice.append(MEDICAL_ADVICE_TEMPLATES[symptom]["advice"])

        return " ".join(advice)
