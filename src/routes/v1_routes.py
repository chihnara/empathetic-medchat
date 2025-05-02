"""
Original v1 routes for the medical chat interface.
"""

from flask import Blueprint, render_template, request, jsonify
from src.models.context_analyzer import ContextAnalyzer
from src.models.empathy_classifier import EmpathyClassifier
import torch
import gc
from typing import Dict, List, Tuple
import random
from collections import deque

# Create blueprint
v1_bp = Blueprint("v1", __name__, url_prefix="/v1")

# Initialize components
print("Initializing v1 components...")
context_analyzer = ContextAnalyzer(medical_model_name="d4data/biomedical-ner-all")
empathy_classifier = EmpathyClassifier(
    model_name="distilbert-base-uncased", device="cpu"
)

# Store conversation states for different sessions
conversation_states = {}


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
        for symptom in medical_context["symptoms"]:
            if not any(
                s["text"] == symptom["text"] for s in self.medical_context["symptoms"]
            ):
                self.medical_context["symptoms"].append(symptom)

        # Update conditions
        for condition in medical_context["conditions"]:
            if not any(
                c["text"] == condition["text"]
                for c in self.medical_context["conditions"]
            ):
                self.medical_context["conditions"].append(condition)

        # Update treatments
        for treatment in medical_context["treatments"]:
            if not any(
                t["text"] == treatment["text"]
                for t in self.medical_context["treatments"]
            ):
                self.medical_context["treatments"].append(treatment)

        # Update medications
        for medication in medical_context["medications"]:
            if not any(
                m["text"] == medication["text"]
                for m in self.medical_context["medications"]
            ):
                self.medical_context["medications"].append(medication)

        # Update emotional context
        if emotional_context["emotions"]:
            self.emotional_context["emotions"].extend(
                [
                    e
                    for e in emotional_context["emotions"]
                    if e not in self.emotional_context["emotions"]
                ]
            )
            self.emotional_context["sentiment"] = emotional_context["sentiment"]
            self.emotional_context["confidence"] = max(
                self.emotional_context["confidence"], emotional_context["confidence"]
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


def generate_response(
    medical_context: Dict,
    emotional_context: Dict,
    empathy_level: str,
    conversation_state: ConversationState,
) -> str:
    """Generate a response based on context and empathy level."""
    # Extract key information
    symptoms = [item["text"] for item in medical_context["symptoms"]]
    conditions = [item["text"] for item in medical_context["conditions"]]
    treatments = [item["text"] for item in medical_context["treatments"]]
    medications = [item["text"] for item in medical_context["medications"]]
    emotions = emotional_context["emotions"]

    # Medical knowledge base (simplified for example)
    medical_knowledge = {
        "fever": {
            "follow_up": "How long have you had the fever? Have you taken your temperature?",
            "advice": "It's important to stay hydrated and rest. If your fever persists for more than 3 days or is above 103°F, please seek medical attention.",
            "related": ["chills", "fatigue", "headache"],
        },
        "headache": {
            "follow_up": "Where is the pain located? Is it throbbing or constant?",
            "advice": "Try resting in a quiet, dark room. Over-the-counter pain relievers may help, but consult a doctor if the pain is severe or persistent.",
            "related": ["migraine", "tension", "sinus"],
        },
        "cough": {
            "follow_up": "Is the cough dry or productive? How long has it been going on?",
            "advice": "Stay hydrated and consider using a humidifier. If the cough persists for more than 2 weeks or is accompanied by difficulty breathing, seek medical attention.",
            "related": ["cold", "flu", "allergies"],
        },
    }

    # Determine empathy level based on context
    if emotions and any(e in ["anxiety", "fear", "pain", "distress"] for e in emotions):
        empathy_level = "high"
    elif "severe" in " ".join(symptoms).lower() or any(
        s in ["pain", "emergency", "urgent"] for s in symptoms
    ):
        empathy_level = "high"
    elif not symptoms and not conditions:
        empathy_level = "low"
    elif conversation_state.history and len(conversation_state.history) > 2:
        # Increase empathy for persistent issues
        empathy_level = "high"
    elif any(e in ["sad", "worried", "concerned"] for e in emotions):
        empathy_level = "medium"

    # Check if we need to ask for more information
    if not symptoms and not conditions:
        return "I understand you're not feeling well. Could you tell me more about your symptoms so I can better help you?"

    # Build response based on context
    response_parts = []

    # 1. Empathetic acknowledgment
    if empathy_level == "high":
        response_parts.append(
            f"I understand how difficult this must be for you. I can see you're experiencing {', '.join(symptoms)}."
        )
    elif empathy_level == "medium":
        response_parts.append(f"I hear you're dealing with {', '.join(symptoms)}.")
    else:
        response_parts.append(f"I see you're experiencing {', '.join(symptoms)}.")

    # 2. Emotional validation
    if emotions:
        emotion_text = ", ".join(emotions)
        if "negative" in emotions:
            response_parts.append(
                f"It's completely understandable to feel {emotion_text} in this situation."
            )
        else:
            response_parts.append(f"I notice you're feeling {emotion_text}.")

    # 3. Medical context and advice
    for symptom in symptoms:
        if symptom.lower() in medical_knowledge:
            knowledge = medical_knowledge[symptom.lower()]
            if not any(
                q in conversation_state.history
                for q in knowledge["follow_up"].split("?")
            ):
                response_parts.append(knowledge["follow_up"])
            else:
                response_parts.append(knowledge["advice"])

    # 4. Follow-up questions based on medical knowledge
    if symptoms and not any("?" in part for part in response_parts):
        primary_symptom = symptoms[0].lower()
        if primary_symptom in medical_knowledge:
            response_parts.append(medical_knowledge[primary_symptom]["follow_up"])

    # 5. Treatment context
    if treatments:
        response_parts.append(
            f"I see you're currently using {', '.join(treatments)}. How has that been working for you?"
        )

    # 6. Medication context
    if medications:
        response_parts.append(
            f"You mentioned taking {', '.join(medications)}. Have you noticed any side effects?"
        )

    # 7. Conversation continuity
    if conversation_state.history:
        last_user_message = conversation_state.history[-1]["user"]
        if "?" in last_user_message and not any("?" in part for part in response_parts):
            response_parts.append(
                "Is there anything specific you'd like to know more about?"
            )

    # Combine response parts
    response = " ".join(response_parts)

    # Ensure response ends with a question if it's a new topic
    if (
        not any("?" in part for part in response_parts)
        and len(conversation_state.history) < 2
    ):
        response += " Could you tell me more about how this started?"

    return response


@v1_bp.route("/")
def home_v1():
    """Render the v1 chat interface."""
    return render_template("index.html")


@v1_bp.route("/chat", methods=["POST"])
def chat_v1():
    """Handle chat messages."""
    data = request.json
    user_input = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    # Initialize or get conversation state for this session
    if session_id not in conversation_states:
        conversation_states[session_id] = ConversationState()
    conversation_state = conversation_states[session_id]

    # Analyze context
    medical_context, emotional_context = context_analyzer.analyze_context(user_input)

    # Update conversation state
    conversation_state.update_context(medical_context, emotional_context)

    # Generate response
    response = generate_response(
        medical_context, emotional_context, "medium", conversation_state
    )

    # Classify empathy of the generated response
    empathy_level, confidence = empathy_classifier.predict(response)

    # Add to history
    conversation_state.add_to_history(user_input, response)

    # Prepare response data
    response_data = {
        "response": response,
        "medical_context": medical_context,
        "emotional_context": emotional_context,
        "empathy_level": empathy_level,
        "empathy_confidence": confidence,
        "conversation_context": conversation_state.get_context_summary(),
    }

    return jsonify(response_data)


@v1_bp.route("/reset", methods=["POST"])
def reset_v1():
    """Reset the conversation state."""
    data = request.json
    session_id = data.get("session_id", "default")

    if session_id in conversation_states:
        del conversation_states[session_id]

    return jsonify({"status": "success"})
