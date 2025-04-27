"""
Response generator for medical dialogue with empathy.
"""

from typing import Dict, List, Tuple
import random


class ResponseGenerator:
    """Generates empathetic medical responses using templates."""

    def __init__(self):
        """Initialize the response generator."""
        self.templates = {
            "high": [
                "I understand that {symptoms} can be very concerning. I hear how {emotions} you're feeling. Let's work together to address this and find the best way to help you feel better.",
                "I can see how {symptoms} is really affecting you, and it's completely natural to feel {emotions}. I'm here to support you and help you manage this situation.",
                "Thank you for sharing about your {symptoms}. It must be challenging to deal with this, especially feeling {emotions}. Let's discuss how we can help you through this.",
            ],
            "medium": [
                "I see you're experiencing {symptoms}. It's understandable to feel {emotions}. Let's look at what we can do to help.",
                "Having {symptoms} can be difficult. I want to help you manage this and address your {emotions} feelings.",
                "Let's talk about your {symptoms} and how we can help you feel better. It's normal to feel {emotions} about this.",
            ],
            "low": [
                "I'll help you with your {symptoms}. We can work on managing this together.",
                "Let's address your {symptoms} and find a solution that works for you.",
                "We'll focus on treating your {symptoms} and improving your condition.",
            ],
        }

    def generate_response(
        self, context: Dict, empathy_level: str, max_length: int = 50
    ) -> str:
        """Generate a response based on context and empathy level."""
        try:
            # Extract context
            medical_context = context.get("medical", {})
            symptoms = [s["text"] for s in medical_context.get("symptoms", [])]
            symptoms_text = "symptoms" if not symptoms else ", ".join(symptoms)

            emotional_context = context.get("emotional", {})
            emotions = emotional_context.get("emotions", [])
            emotions_text = "concerned" if not emotions else ", ".join(emotions)

            # Select template
            templates = self.templates.get(
                empathy_level.lower(), self.templates["medium"]
            )
            template = random.choice(templates)

            # Generate response
            response = template.format(symptoms=symptoms_text, emotions=emotions_text)

            return response

        except Exception as e:
            print(f"Error in response generation: {str(e)}")
            return "I apologize, but I'm having trouble generating a response at the moment."
