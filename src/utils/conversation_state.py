"""
Conversation state management utilities.
"""

from collections import deque
from typing import Dict, List, Tuple


class ConversationState:
    """Maintains conversation state and history."""

    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        self.medical_context = {
            "symptoms": [],
            "conditions": [],
            "treatments": [],
            "medications": [],
            "confidence": 0.0,
        }
        self.emotional_context = {
            "sentiment": "neutral",
            "emotions": [],
            "confidence": 0.0,
        }

    def update_context(self, medical_context: Dict, emotional_context: Dict):
        """Update the current context with new information."""
        # Update symptoms
        for symptom in medical_context.get("symptoms", []):
            if not any(
                s["text"] == symptom["text"] for s in self.medical_context["symptoms"]
            ):
                self.medical_context["symptoms"].append(symptom)

        # Update conditions
        for condition in medical_context.get("conditions", []):
            if not any(
                c["text"] == condition["text"]
                for c in self.medical_context["conditions"]
            ):
                self.medical_context["conditions"].append(condition)

        # Update treatments
        for treatment in medical_context.get("treatments", []):
            if not any(
                t["text"] == treatment["text"]
                for t in self.medical_context["treatments"]
            ):
                self.medical_context["treatments"].append(treatment)

        # Update medications
        for medication in medical_context.get("medications", []):
            if not any(
                m["text"] == medication["text"]
                for m in self.medical_context["medications"]
            ):
                self.medical_context["medications"].append(medication)

        # Update emotional context
        if emotional_context.get("emotions"):
            self.emotional_context["emotions"].extend(
                [
                    e
                    for e in emotional_context["emotions"]
                    if e not in self.emotional_context["emotions"]
                ]
            )
            self.emotional_context["sentiment"] = emotional_context.get(
                "sentiment", "neutral"
            )
            self.emotional_context["confidence"] = max(
                self.emotional_context["confidence"],
                emotional_context.get("confidence", 0.0),
            )

    def add_to_history(self, user_input: str, response: str):
        """Add a conversation turn to history."""
        self.history.append(
            {
                "user": user_input,
                "assistant": response,
                "medical_context": self.medical_context.copy(),
                "emotional_context": self.emotional_context.copy(),
            }
        )

    def get_context_summary(self) -> str:
        """Get a summary of the current context."""
        summary = []

        if self.medical_context["symptoms"]:
            symptoms = ", ".join(s["text"] for s in self.medical_context["symptoms"])
            summary.append(f"Symptoms: {symptoms}")

        if self.medical_context["conditions"]:
            conditions = ", ".join(
                c["text"] for c in self.medical_context["conditions"]
            )
            summary.append(f"Conditions: {conditions}")

        if self.emotional_context["emotions"]:
            emotions = ", ".join(self.emotional_context["emotions"])
            summary.append(f"Emotions: {emotions}")

        return " | ".join(summary) if summary else "No specific context"
