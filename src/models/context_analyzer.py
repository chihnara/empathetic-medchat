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
        medical_model_name: str = "models/medical_ner",
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

                        # Skip tokens that start with ## or are too short
                        if text.startswith("##") or len(text) < 2:
                            continue

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

            # Additional processing for common medical terms
            text_lower = text.lower()

            # Check for common symptom patterns
            symptom_patterns = [
                ("headache", "head pain"),
                ("cold", "common cold"),
                ("fever", "high temperature"),
                ("cough", "coughing"),
                ("sore throat", "throat pain"),
                ("runny nose", "nasal congestion"),
                ("nausea", "feeling sick"),
                ("dizziness", "lightheaded"),
                ("fatigue", "tiredness"),
                ("insomnia", "trouble sleeping"),
                ("back pain", "backache"),
                ("chest pain", "chest discomfort"),
                ("shortness of breath", "difficulty breathing"),
                ("muscle pain", "muscle ache"),
                ("joint pain", "joint ache"),
                ("bloating", "abdominal bloating"),
                ("cramps", "abdominal cramps"),
                ("stomach pain", "abdominal pain"),
                ("indigestion", "upset stomach"),
                ("constipation", "bowel problems"),
                ("diarrhea", "loose stools"),
                ("vomiting", "throwing up"),
                ("loss of appetite", "decreased appetite"),
                ("weight loss", "unintended weight loss"),
                ("weight gain", "unintended weight gain"),
            ]

            # First check for exact matches
            for pattern, alternative in symptom_patterns:
                if pattern in text_lower or alternative in text_lower:
                    if not any(
                        pattern in item["text"].lower() for item in context["symptoms"]
                    ):
                        context["symptoms"].append({"text": pattern, "confidence": 0.6})

            # Then check for severity indicators
            severity_indicators = [
                "severe",
                "bad",
                "terrible",
                "awful",
                "extreme",
                "intense",
                "chronic",
                "persistent",
            ]
            for indicator in severity_indicators:
                if indicator in text_lower:
                    # Find the closest symptom to the severity indicator
                    words = text_lower.split()
                    for i, word in enumerate(words):
                        if word == indicator and i + 1 < len(words):
                            next_word = words[i + 1]
                            # Only add severity if we find a matching symptom pattern
                            for pattern, _ in symptom_patterns:
                                if next_word in pattern:
                                    if not any(
                                        pattern in item["text"].lower()
                                        for item in context["symptoms"]
                                    ):
                                        context["symptoms"].append(
                                            {
                                                "text": f"{indicator} {pattern}",
                                                "confidence": 0.7,
                                            }
                                        )
                                    break

            # Remove any symptoms that were incorrectly inferred
            context["symptoms"] = [
                symptom
                for symptom in context["symptoms"]
                if symptom["text"].lower() in text_lower
                or any(
                    pattern in text_lower
                    for pattern, _ in symptom_patterns
                    if pattern in symptom["text"].lower()
                )
            ]

            # Clean up any remaining strange tokens
            context["symptoms"] = [
                symptom
                for symptom in context["symptoms"]
                if not symptom["text"].startswith("##") and len(symptom["text"]) > 2
            ]

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
            "cold",
            "flu",
            "sneeze",
            "runny nose",
            "congestion",
            "sore throat",
            "muscle ache",
            "joint pain",
            "back pain",
            "chest pain",
            "shortness of breath",
            "difficulty breathing",
            "wheezing",
            "insomnia",
            "sleep problems",
            "trouble sleeping",
            "loss of appetite",
            "nausea",
            "vomiting",
            "diarrhea",
            "constipation",
            "bloating",
            "abdominal pain",
        ]

        # Common conditions keywords
        condition_keywords = [
            "diabetes",
            "hypertension",
            "high blood pressure",
            "low blood pressure",
            "asthma",
            "arthritis",
            "depression",
            "anxiety",
            "infection",
            "injury",
            "cold",
            "flu",
            "pneumonia",
            "bronchitis",
            "sinusitis",
            "migraine",
            "allergy",
            "allergic",
            "chronic",
            "acute",
            "respiratory",
            "cardiovascular",
            "gastrointestinal",
            "neurological",
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
            "prescription",
            "over-the-counter",
            "OTC",
            "antibiotics",
            "pain relievers",
            "anti-inflammatory",
            "antihistamines",
            "decongestants",
            "cough medicine",
            "sleep aids",
        ]

        # Common medication keywords
        medication_keywords = [
            "aspirin",
            "ibuprofen",
            "paracetamol",
            "acetaminophen",
            "antibiotic",
            "insulin",
            "antidepressant",
            "antihistamine",
            "steroid",
            "painkiller",
            "tylenol",
            "advil",
            "motrin",
            "benadryl",
            "claritin",
            "zyrtec",
            "sudafed",
            "nyquil",
            "dayquil",
            "robitussin",
            "mucinex",
        ]

        # Process the text for each category
        for category, keywords in [
            ("symptoms", symptom_keywords),
            ("conditions", condition_keywords),
            ("treatments", treatment_keywords),
            ("medications", medication_keywords),
        ]:
            existing_texts = {item["text"].lower() for item in context[category]}

            # Check for multi-word phrases first
            for keyword in sorted(keywords, key=len, reverse=True):
                if keyword in existing_texts:
                    continue

                # Check if the keyword is in the text
                if keyword.lower() in " ".join(existing_texts).lower():
                    context[category].append({"text": keyword, "confidence": 0.5})
                # Check for partial matches
                elif any(keyword.lower() in text.lower() for text in existing_texts):
                    context[category].append({"text": keyword, "confidence": 0.4})
                # Check for word boundaries
                elif any(
                    keyword.lower() in f" {text.lower()} " for text in existing_texts
                ):
                    context[category].append({"text": keyword, "confidence": 0.3})

    def _process_emotional_context(self, text: str) -> Dict:
        """Process emotional context from text."""
        # Simple emotional analysis based on keywords
        emotional_keywords = {
            "positive": [
                "good",
                "better",
                "improve",
                "help",
                "relief",
                "improving",
                "helped",
                "working",
                "effective",
            ],
            "negative": [
                "pain",
                "hurt",
                "worse",
                "bad",
                "uncomfortable",
                "severe",
                "terrible",
                "extreme",
                "intense",
                "persistent",
                "constant",
                "chronic",
                "awful",
            ],
            "concern": [
                "worry",
                "concern",
                "anxious",
                "nervous",
                "scared",
                "afraid",
                "stressed",
                "unsure",
                "uncertain",
            ],
            "gratitude": ["thank", "appreciate", "grateful", "thanks"],
        }

        context = {"sentiment": "neutral", "emotions": [], "confidence": 0.0}
        text_lower = text.lower()

        # First check for direct emotional keywords
        detected_emotions = []
        for emotion, keywords in emotional_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_emotions.append(emotion)
                context["confidence"] = 0.7

        # If no direct emotions detected, check medical context
        if not detected_emotions:
            # Check for severity indicators with symptoms
            severity_words = [
                "severe",
                "terrible",
                "extreme",
                "intense",
                "persistent",
                "constant",
            ]
            symptom_words = [
                "fever",
                "cough",
                "headache",
                "pain",
                "ache",
                "dizziness",
                "nausea",
                "fatigue",
                "infection",
                "sick",
                "ill",
            ]

            has_severity = any(word in text_lower for word in severity_words)
            has_symptoms = any(word in text_lower for word in symptom_words)

            if has_severity and has_symptoms:
                detected_emotions.append("negative")
                context["confidence"] = 0.6
            elif has_symptoms:
                detected_emotions.append("concern")
                context["confidence"] = 0.5

            # Check for duration indicators
            duration_words = [
                "week",
                "days",
                "months",
                "long",
                "still",
                "keeps",
                "continuous",
            ]
            if any(word in text_lower for word in duration_words) and has_symptoms:
                if "concern" not in detected_emotions:
                    detected_emotions.append("concern")
                context["confidence"] = 0.6

        # Set emotions and sentiment
        if detected_emotions:
            context["emotions"] = detected_emotions
            context["sentiment"] = (
                "positive" if "positive" in detected_emotions else "negative"
            )
        else:
            # Default for medical conversations
            context["emotions"] = ["concern"]
            context["sentiment"] = "neutral"
            context["confidence"] = 0.4

        return context
