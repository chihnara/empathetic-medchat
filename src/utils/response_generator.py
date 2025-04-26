"""
Response generation utilities for medical dialogue.
"""

from typing import Dict, List, Optional
from ..models.dialogue_model import DialogueGenerator
from ..utils.context_analyzer import ContextAnalyzer
from ..config.config import EMPATHY_TEMPLATES, MEDICAL_ADVICE_TEMPLATES


class ResponseGenerator:
    """Generates empathetic medical responses."""

    def __init__(self, model_path: str):
        """Initialize response generator."""
        self.model = DialogueGenerator.load(model_path)
        self.context_analyzer = ContextAnalyzer()

    def generate_response(
        self,
        query: str,
        empathy_level: str,
        max_length: int = 200,
        temperature: float = 0.7,
    ) -> str:
        """Generate empathetic medical response."""
        # Analyze context
        emotions = self.context_analyzer.extract_emotional_context(query)
        symptoms = self.context_analyzer.extract_medical_context(query)

        # Get emotional intensity
        intensity = self.context_analyzer.get_emotional_intensity(emotions)

        # Get medical advice
        medical_advice = self.context_analyzer.get_medical_advice(symptoms)

        # Select empathy template
        template = EMPATHY_TEMPLATES[empathy_level][intensity]

        # Format response
        response = template.format(
            medical_advice=medical_advice,
            symptoms=", ".join(symptoms) if symptoms else "your symptoms",
        )

        # Generate response using model
        generated = self.model.generate(
            prompt=response, max_length=max_length, temperature=temperature
        )

        return generated
