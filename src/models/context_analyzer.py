"""
Context analysis for medical dialogue.
"""

from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from functools import lru_cache


class ContextAnalyzer:
    """Analyzes medical and emotional context from dialogue."""

    def __init__(
        self,
        medical_model_name: str = "src/models/medical_ner",
    ):
        """Initialize the context analyzer with medical NER model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.medical_model_name = medical_model_name

        # Initialize NER pipeline with optimized settings
        self.ner_pipeline = pipeline(
            "ner",
            model=self.medical_model_name,
            tokenizer=self.medical_model_name,
            device=0 if self.device == "cuda" else -1,
            aggregation_strategy="simple",
            batch_size=8,
        )

        # Medical entity types mapping
        self.medical_entity_types = {
            "SYMPTOM": "symptoms",
            "CONDITION": "conditions",
            "TREATMENT": "treatments",
            "DRUG": "medications",
        }

    @lru_cache(maxsize=100)
    def analyze_context(self, text: str) -> Tuple[Dict, Dict]:
        """Analyze medical and emotional context from text."""
        # Process medical entities
        medical_context = self._process_medical_entities(text)

        # Process emotional context
        emotional_context = self._process_emotional_context(text)

        return medical_context, emotional_context

    def _process_medical_entities(self, text: str) -> Dict:
        """Process medical entities from text."""
        try:
            # Get NER predictions
            entities = self.ner_pipeline(text)

            # Initialize context dictionary
            context = {
                "symptoms": [],
                "conditions": [],
                "treatments": [],
                "medications": [],
                "confidence": 0.0,
            }

            # Process entities
            total_score = 0.0
            entity_count = 0

            if entities:
                for entity in entities:
                    entity_type = entity.get("entity_group", "")
                    if entity_type in self.medical_entity_types:
                        category = self.medical_entity_types[entity_type]
                        text = entity.get("word", "").strip()
                        score = float(entity.get("score", 0.0))

                        if score > 0.3:  # Lower threshold for better recall
                            context[category].append(
                                {"text": text, "confidence": score}
                            )
                            total_score += score
                            entity_count += 1

                # Calculate overall confidence
                if entity_count > 0:
                    context["confidence"] = total_score / entity_count

            # Post-process entities
            self._post_process_entities(context)

            return context

        except Exception as e:
            print(f"Error processing medical entities: {str(e)}")
            return {
                "symptoms": [],
                "conditions": [],
                "treatments": [],
                "medications": [],
                "confidence": 0.0,
            }

    def _post_process_entities(self, context: Dict) -> None:
        """Post-process entities to improve accuracy."""
        # Common symptoms keywords
        symptom_keywords = [
            "pain",
            "ache",
            "discomfort",
            "headache",
            "dizziness",
            "nausea",
            "fatigue",
            "fever",
            "cough",
            "sore",
            "swelling",
            "rash",
        ]

        # Common conditions keywords
        condition_keywords = [
            "diabetes",
            "hypertension",
            "pressure",
            "chronic",
            "asthma",
            "arthritis",
            "depression",
            "anxiety",
            "infection",
            "injury",
        ]

        # Common treatments keywords
        treatment_keywords = [
            "medication",
            "treatment",
            "therapy",
            "surgery",
            "exercise",
            "diet",
            "rest",
            "rehabilitation",
            "physical therapy",
        ]

        # Common medication keywords
        medication_keywords = [
            "aspirin",
            "ibuprofen",
            "paracetamol",
            "antibiotic",
            "insulin",
            "antidepressant",
            "antihistamine",
            "steroid",
            "painkiller",
        ]

        # Check text for common keywords
        for category, keywords in [
            ("symptoms", symptom_keywords),
            ("conditions", condition_keywords),
            ("treatments", treatment_keywords),
            ("medications", medication_keywords),
        ]:
            existing_texts = {item["text"].lower() for item in context[category]}

            for keyword in keywords:
                if keyword in existing_texts:
                    continue

                if keyword in " ".join(existing_texts).lower():
                    context[category].append({"text": keyword, "confidence": 0.5})

    def _process_emotional_context(self, text: str) -> Dict:
        """Process emotional context from text."""
        # Simple emotional analysis based on keywords
        emotional_keywords = {
            "positive": ["good", "better", "improve", "help", "relief"],
            "negative": ["pain", "hurt", "worse", "bad", "uncomfortable"],
            "concern": ["worry", "concern", "anxious", "nervous"],
            "gratitude": ["thank", "appreciate", "grateful"],
        }

        context = {"sentiment": "neutral", "emotions": [], "confidence": 0.0}

        text_lower = text.lower()
        detected_emotions = []

        for emotion, keywords in emotional_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_emotions.append(emotion)

        if detected_emotions:
            context["emotions"] = detected_emotions
            context["confidence"] = 0.7  # High confidence for keyword-based detection
            context["sentiment"] = (
                "positive" if "positive" in detected_emotions else "negative"
            )

        return context
