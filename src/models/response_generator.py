"""
Response generator for medical dialogue with empathy.
"""

from typing import Dict, List, Tuple
import random
import logging

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates empathetic medical responses using a template-based approach."""

    def __init__(self):
        """Initialize the response generator."""
        try:
            # Template components for response generation
            self.acknowledgments = {
                "high": [
                    "I understand how concerning {symptoms} can be.",
                    "I hear how {symptoms} is affecting you.",
                    "It must be difficult dealing with {symptoms}.",
                ],
                "medium": [
                    "I see you're experiencing {symptoms}.",
                    "Thank you for sharing about your {symptoms}.",
                    "Let's address your {symptoms}.",
                ],
                "low": [
                    "I'll help you with your {symptoms}.",
                    "Let's look at your {symptoms}.",
                    "We'll address your {symptoms}.",
                ],
            }

            self.emotional_support = {
                "high": [
                    "It's completely natural to feel {emotions}.",
                    "Your feelings of {emotions} are valid.",
                    "Many people feel {emotions} in similar situations.",
                ],
                "medium": [
                    "I understand you're feeling {emotions}.",
                    "It's common to feel {emotions}.",
                    "Let's work through these {emotions} feelings.",
                ],
                "low": [
                    "We'll help you manage these symptoms.",
                    "Let's focus on addressing your concerns.",
                    "We can work on improving your condition.",
                ],
            }

            self.action_statements = {
                "high": [
                    "Let's work together to find the best way to help you feel better. I'll guide you through understanding your symptoms and exploring treatment options.",
                    "I'm here to support you and help you manage this situation. We can discuss different approaches and find what works best for you.",
                    "We'll take this one step at a time and find what works best for you. I'll help you understand your symptoms and develop a management plan.",
                ],
                "medium": [
                    "Let's look at what we can do to help. We can explore different treatment options and find an effective approach.",
                    "We can work on managing these symptoms together. I'll explain the available treatments and help you make informed decisions.",
                    "There are several approaches we can try. We'll find the most effective way to address your symptoms.",
                ],
                "low": [
                    "Here's what we can do to address this. We'll focus on evidence-based treatments and symptom management.",
                    "Let's focus on treating these symptoms effectively. I'll outline the recommended medical approaches.",
                    "We'll develop a plan to manage this condition using proven medical treatments.",
                ],
            }

            # Medical context templates
            self.medical_context = {
                "headache": [
                    "Headaches can vary in intensity and type, so it's important to understand your specific symptoms.",
                    "There are various factors that can trigger headaches, including stress, tension, or underlying conditions.",
                    "We can work on both immediate relief and long-term management strategies.",
                ],
                "pain": [
                    "Pain management involves understanding both the cause and finding effective relief methods.",
                    "We can explore both medication and non-medication approaches to help manage your pain.",
                    "It's important to address both the physical and emotional aspects of chronic pain.",
                ],
                "blood pressure": [
                    "Blood pressure management involves lifestyle changes and possibly medication.",
                    "We can work on developing a comprehensive plan to maintain healthy blood pressure levels.",
                    "Regular monitoring and proper medication management are key to blood pressure control.",
                ],
                "diabetes": [
                    "Diabetes management involves balancing medication, diet, and lifestyle factors.",
                    "We can work on developing a sustainable plan for managing your blood sugar levels.",
                    "Understanding your medication schedule and monitoring requirements is crucial.",
                ],
                "sleep": [
                    "Sleep problems can significantly impact your health and well-being.",
                    "We can explore various strategies to improve your sleep quality.",
                    "Both behavioral changes and medical approaches can help with sleep issues.",
                ],
            }

            logger.info("Response generator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing response generator: {str(e)}")
            raise

    def _format_context(
        self, medical_context: Dict, emotional_context: Dict
    ) -> Tuple[str, str, List[str]]:
        """Format medical and emotional context into components."""
        try:
            # Extract symptoms
            symptoms = []
            for entity_type, entities in medical_context.items():
                if entity_type == "confidence":
                    continue
                if isinstance(entities, list) and entities:
                    symptoms.extend(
                        [
                            e["text"]
                            for e in entities
                            if isinstance(e, dict) and "text" in e
                        ]
                    )

            # Extract emotions
            emotions = emotional_context.get("emotions", [])
            if not isinstance(emotions, list):
                emotions = []

            # Format for template
            symptoms_text = "these symptoms" if not symptoms else ", ".join(symptoms)
            emotions_text = "concerned" if not emotions else ", ".join(emotions)

            return symptoms_text, emotions_text, symptoms

        except Exception as e:
            logger.error(f"Error formatting context: {str(e)}")
            return "your symptoms", "concerned", []

    def _get_medical_context(self, symptoms: List[str]) -> str:
        """Get relevant medical context based on symptoms."""
        context_statements = []

        # Look for keyword matches in symptoms
        for symptom in symptoms:
            for keyword, statements in self.medical_context.items():
                if keyword in symptom.lower():
                    context_statements.append(random.choice(statements))
                    break

        if context_statements:
            return " " + random.choice(context_statements)
        return ""

    def generate_response(
        self,
        query: str,
        medical_context: Dict,
        emotional_context: Dict,
        empathy_level: str,
    ) -> str:
        """Generate a response based on query and context with specified empathy level."""
        try:
            # Format context
            symptoms_text, emotions_text, symptoms = self._format_context(
                medical_context, emotional_context
            )

            # Select template components based on empathy level
            level = empathy_level.lower()

            # Build response from templates
            acknowledgment = random.choice(self.acknowledgments[level]).format(
                symptoms=symptoms_text
            )

            emotional = ""
            if emotions_text and level != "low":
                emotional = " " + random.choice(self.emotional_support[level]).format(
                    emotions=emotions_text
                )

            action = " " + random.choice(self.action_statements[level])

            # Add relevant medical context if available
            medical_context = self._get_medical_context(symptoms)

            # Combine all components
            response = acknowledgment + emotional + medical_context + action

            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response at the moment."
