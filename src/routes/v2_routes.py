"""
V2 routes for the enhanced medical chat interface.
"""

from flask import Blueprint, render_template, request, jsonify
from src.models.v2.enhanced_empathy_classifier import EnhancedEmpathyClassifier
from src.models.v2.enhanced_paraphraser import EnhancedParaphraser
from src.models.v2.medical_knowledge_base import MedicalKnowledgeBase
from src.models.context_analyzer import ContextAnalyzer
from collections import deque
import gc
import torch

# Initialize v2 components
print("Initializing v2 components...")
context_analyzer = ContextAnalyzer(medical_model_name="d4data/biomedical-ner-all")
empathy_classifier = EnhancedEmpathyClassifier(
    model_name="distilbert-base-uncased", device="cpu"
)
paraphraser = EnhancedParaphraser()
knowledge_base = MedicalKnowledgeBase()

# Create blueprint
v2_bp = Blueprint("v2", __name__, url_prefix="/v2")

# Store conversation states for different sessions
conversation_states_v2 = {}


class ConversationStateV2:
    """Enhanced conversation state management."""

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
            "emotions": [],
            "empathy_level": "medium",
            "confidence": 0.5,
        }

    def update_context(self, medical_context: dict, emotional_context: dict):
        """Update conversation context with new information."""
        # Update medical context
        for category in ["symptoms", "conditions", "treatments", "medications"]:
            for item in medical_context.get(category, []):
                if not any(
                    existing["text"] == item["text"]
                    for existing in self.medical_context[category]
                ):
                    self.medical_context[category].append(item)

        # Update medical confidence
        self.medical_context["confidence"] = medical_context.get("confidence", 0.0)

        # Update emotional context
        if emotional_context.get("emotions"):
            self.emotional_context["emotions"] = emotional_context["emotions"]
            self.emotional_context["empathy_level"] = emotional_context["empathy_level"]
            self.emotional_context["confidence"] = emotional_context["confidence"]

    def add_to_history(self, user_input: str, response: str):
        """Add conversation turn to history."""
        self.history.append(
            {
                "user": user_input,
                "assistant": response,
                "medical_context": self.medical_context.copy(),
                "emotional_context": self.emotional_context.copy(),
            }
        )

    def get_context_summary(self) -> str:
        """Get a summary of current context."""
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
            summary.append(
                f"Emotions: {emotions} (Level: {self.emotional_context['empathy_level']})"
            )

        return " | ".join(summary) if summary else "No specific context"


def generate_enhanced_response(
    medical_context: dict,
    emotional_context: dict,
    conversation_state: ConversationStateV2,
) -> str:
    """Generate enhanced response based on context."""
    try:
        # Extract key information
        symptoms = [item["text"] for item in medical_context["symptoms"]]
        conditions = [item["text"] for item in medical_context["conditions"]]
        treatments = [item["text"] for item in medical_context["treatments"]]
        medications = [item["text"] for item in medical_context["medications"]]

        # Get emotional style
        empathy_level = emotional_context["empathy_level"]
        emotions = emotional_context.get("emotions", [])

        # Build response parts
        response_parts = []

        # Check conversation history for new vs. continuing symptoms
        existing_symptoms = set()
        new_symptoms = set(symptoms)
        if conversation_state.history:
            last_context = conversation_state.history[-1]["medical_context"]
            existing_symptoms = {s["text"] for s in last_context["symptoms"]}
            new_symptoms = set(symptoms) - existing_symptoms

        # 1. Empathetic acknowledgment based on context
        if new_symptoms:  # Only acknowledge new symptoms
            if empathy_level == "high":
                response_parts.append(
                    f"I understand that you're also experiencing {', '.join(new_symptoms)}, which must be quite concerning."
                )
            elif empathy_level == "medium":
                response_parts.append(
                    f"I see you're also having {', '.join(new_symptoms)}."
                )
            else:
                response_parts.append(f"You've mentioned {', '.join(new_symptoms)}.")
        elif not conversation_state.history:  # First message
            if symptoms:
                if empathy_level == "high":
                    response_parts.append(
                        f"I understand how concerning it must be to experience {', '.join(symptoms)}."
                    )
                elif empathy_level == "medium":
                    response_parts.append(
                        f"I hear that you're experiencing {', '.join(symptoms)}."
                    )
                else:
                    response_parts.append(
                        f"You mentioned experiencing {', '.join(symptoms)}."
                    )

        # 2. Emotional validation (only if emotions indicate distress)
        if emotions:
            distress_emotions = [
                e for e in emotions if e in ["distress", "anxiety", "concern"]
            ]
            if distress_emotions:
                response_parts.append(
                    f"It's completely understandable to feel {', '.join(distress_emotions)} about these symptoms."
                )

        # 3. Medical context integration
        if conditions:
            # Query knowledge base for condition information
            condition_info = knowledge_base.query_condition(conditions[0])
            if condition_info["treatments"]:
                response_parts.append(
                    f"For {conditions[0]}, common treatments include {', '.join(condition_info['treatments'])}."
                )

        # 4. Treatment and medication context
        if treatments and not any(
            "treatment" in part.lower() for part in response_parts
        ):
            response_parts.append(
                f"I see you're currently using {', '.join(treatments)}. How has that been working for you?"
            )
        if medications and not any(
            "medication" in part.lower() for part in response_parts
        ):
            response_parts.append(
                f"Regarding {', '.join(medications)}, have you noticed any side effects?"
            )

        # 5. Follow-up questions based on context
        if not symptoms and not conditions:
            response_parts.append(
                "Could you tell me more about what symptoms you're experiencing? This will help me provide better guidance."
            )
        elif not conversation_state.history:  # Only ask timing for first message
            response_parts.append(
                "When did these symptoms first start? Have they been getting worse?"
            )
        elif new_symptoms:  # Ask about new symptoms
            response_parts.append(
                f"When did you start noticing {', '.join(new_symptoms)}? Are they related to your previous symptoms?"
            )
        elif empathy_level == "high" and not any(
            "?" in part for part in response_parts
        ):
            response_parts.append(
                "How severe are your symptoms right now compared to when they started?"
            )
        elif not any(
            "?" in part for part in response_parts
        ):  # Ensure we have a question
            response_parts.append(
                "How are these symptoms affecting your daily activities?"
            )

        # Combine response parts
        response = " ".join(response_parts)

        # Enhance response with paraphrasing if we have symptoms
        if (
            symptoms and len(response_parts) > 1
        ):  # Only paraphrase substantive responses
            response = paraphraser.paraphrase(
                response, context={"symptoms": symptoms}, style=empathy_level
            )

        return response

    except Exception as e:
        print(f"Error generating enhanced response: {str(e)}")
        return "I apologize, but I'm having trouble processing that. Could you rephrase your concern?"


def analyze_medical_context(text: str) -> dict:
    """Analyze medical context from text."""
    try:
        # Get medical context using the correct method
        medical_context, _ = context_analyzer.analyze_context(text)
        return medical_context
    except Exception as e:
        print(f"Error in medical context analysis: {str(e)}")
        return {
            "symptoms": [],
            "conditions": [],
            "treatments": [],
            "medications": [],
            "confidence": 0.0,
        }


@v2_bp.route("/")
def home_v2():
    """Render v2 home page."""
    return render_template("chat_v2.html")


@v2_bp.route("/chat", methods=["POST"])
def chat_v2():
    """Handle chat messages for v2."""
    try:
        data = request.json
        user_input = data.get("message", "").strip()
        session_id = data.get("session_id", "default")

        # Initialize or get conversation state
        if session_id not in conversation_states_v2:
            conversation_states_v2[session_id] = ConversationStateV2()
        conversation_state = conversation_states_v2[session_id]

        # Analyze medical context
        medical_context = analyze_medical_context(user_input)

        # Analyze emotional context
        emotional_result = empathy_classifier.predict_multilevel(user_input)

        # Structure emotional context
        emotional_context = {
            "emotions": emotional_result["emotions"],
            "empathy_level": emotional_result["empathy_level"],
            "confidence": emotional_result["overall_confidence"],
        }

        # Update conversation state
        conversation_state.update_context(medical_context, emotional_context)

        # Generate enhanced response
        response = generate_enhanced_response(
            medical_context, emotional_context, conversation_state
        )

        # Add to conversation history
        conversation_state.add_to_history(user_input, response)

        # Prepare response data
        response_data = {
            "response": response,
            "medical_context": {
                "symptoms": medical_context["symptoms"],
                "conditions": medical_context["conditions"],
                "treatments": medical_context["treatments"],
                "medications": medical_context["medications"],
            },
            "emotional_context": {
                "emotions": emotional_result["emotions"],
                "empathy_level": emotional_result["empathy_level"],
                "confidence": emotional_result["overall_confidence"],
            },
            "conversation_context": conversation_state.get_context_summary(),
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"Error in chat_v2: {str(e)}")
        return jsonify(
            {
                "response": "I apologize, but I'm having trouble processing your message. Could you try rephrasing it?",
                "medical_context": {
                    "symptoms": [],
                    "conditions": [],
                    "treatments": [],
                    "medications": [],
                },
                "emotional_context": {
                    "emotions": ["neutral"],
                    "empathy_level": "medium",
                    "confidence": 0.5,
                },
                "conversation_context": "Error processing context",
            }
        )


@v2_bp.route("/reset", methods=["POST"])
def reset_v2():
    """Reset v2 conversation."""
    try:
        data = request.json
        session_id = data.get("session_id", "default")

        if session_id in conversation_states_v2:
            del conversation_states_v2[session_id]

        return jsonify({"status": "success"})

    except Exception as e:
        print(f"Error in reset_v2: {str(e)}")
        return jsonify({"status": "error"}), 500
