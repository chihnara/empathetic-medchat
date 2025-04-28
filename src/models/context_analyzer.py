"""
Context analysis for medical dialogue.
"""

from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from functools import lru_cache
import logging
import re

logger = logging.getLogger(__name__)


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

        try:
            # Initialize sentiment analysis pipeline with better model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="finiteautomata/bertweet-base-sentiment-analysis",
                return_all_scores=True,
            )

            # Emotion keywords for better detection
            self.emotion_keywords = {
                "positive": [
                    "better",
                    "improved",
                    "relieved",
                    "grateful",
                    "thankful",
                    "hopeful",
                    "optimistic",
                ],
                "negative": [
                    "worried",
                    "anxious",
                    "scared",
                    "fear",
                    "pain",
                    "hurt",
                    "suffering",
                    "hopeless",
                    "overwhelmed",
                    "frustrated",
                    "depressed",
                    "stressed",
                    "terrified",
                    "desperate",
                ],
                "neutral": [
                    "curious",
                    "wondering",
                    "thinking",
                    "considering",
                    "planning",
                ],
            }

            # Common symptom patterns and their variations
            self.symptom_patterns = {
                r"back\s+pain": "back pain",
                r"headache(s)?": "headache",
                r"dizziness": "dizziness",
                r"nausea": "nausea",
                r"fatigue": "fatigue",
                r"fever": "fever",
                r"cough": "cough",
                r"shortness\s+of\s+breath": "shortness of breath",
                r"chest\s+pain": "chest pain",
                r"joint\s+pain": "joint pain",
                r"muscle\s+pain": "muscle pain",
                r"stomach\s+pain": "stomach pain",
                r"abdominal\s+pain": "abdominal pain",
                r"high\s+blood\s+pressure": "high blood pressure",
                r"low\s+blood\s+pressure": "low blood pressure",
                r"diabetes": "diabetes",
                r"insomnia": "insomnia",
                r"anxiety": "anxiety",
                r"depression": "depression",
            }

            logger.info("Context analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing context analyzer: {str(e)}")
            raise

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
        try:
            # Get sentiment scores
            sentiment_results = self.sentiment_analyzer(text)

            # Extract emotions from text
            detected_emotions = []
            text_lower = text.lower()

            # Check for emotion keywords
            for emotion_type, keywords in self.emotion_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        detected_emotions.append(emotion_type)

            # Determine dominant sentiment
            sentiment_scores = sentiment_results[0]
            max_score = max(sentiment_scores, key=lambda x: x["score"])
            sentiment = max_score["label"].lower()

            # If we found emotion keywords, use them to adjust sentiment
            if detected_emotions:
                if "negative" in detected_emotions:
                    sentiment = "negative"
                elif "positive" in detected_emotions:
                    sentiment = "positive"

            # Calculate confidence based on both sentiment and emotion detection
            confidence = max_score["score"]
            if detected_emotions:
                confidence = max(
                    confidence, 0.7
                )  # Boost confidence if emotions detected

            return {
                "sentiment": sentiment,
                "emotions": list(set(detected_emotions)),  # Remove duplicates
                "confidence": confidence,
            }
        except Exception as e:
            logger.error(f"Error analyzing emotional context: {str(e)}")
            return {"sentiment": "neutral", "emotions": [], "confidence": 0.0}
