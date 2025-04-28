"""
Interactive chat interface for MEDICOD implementation.
"""

import os
import torch
import gc
from typing import Dict, List, Tuple
from src.models.context_analyzer import ContextAnalyzer
from src.models.empathy_classifier import EmpathyClassifier
import time
import random
from collections import deque


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
            "confidence": 0.0
        }
        self.emotional_context = {
            "sentiment": "neutral",
            "emotions": [],
            "confidence": 0.0
        }
    
    def update_context(self, medical_context: Dict, emotional_context: Dict):
        """Update the current context with new information."""
        # Update symptoms
        for symptom in medical_context["symptoms"]:
            if not any(s["text"] == symptom["text"] for s in self.medical_context["symptoms"]):
                self.medical_context["symptoms"].append(symptom)
        
        # Update conditions
        for condition in medical_context["conditions"]:
            if not any(c["text"] == condition["text"] for c in self.medical_context["conditions"]):
                self.medical_context["conditions"].append(condition)
        
        # Update treatments
        for treatment in medical_context["treatments"]:
            if not any(t["text"] == treatment["text"] for t in self.medical_context["treatments"]):
                self.medical_context["treatments"].append(treatment)
        
        # Update medications
        for medication in medical_context["medications"]:
            if not any(m["text"] == medication["text"] for m in self.medical_context["medications"]):
                self.medical_context["medications"].append(medication)
        
        # Update emotional context
        if emotional_context["emotions"]:
            self.emotional_context["emotions"].extend(
                [e for e in emotional_context["emotions"] if e not in self.emotional_context["emotions"]]
            )
            self.emotional_context["sentiment"] = emotional_context["sentiment"]
            self.emotional_context["confidence"] = max(
                self.emotional_context["confidence"],
                emotional_context["confidence"]
            )
    
    def add_to_history(self, user_input: str, response: str):
        """Add a conversation turn to history."""
        self.history.append({
            "user": user_input,
            "assistant": response,
            "medical_context": self.medical_context.copy(),
            "emotional_context": self.emotional_context.copy()
        })
    
    def get_context_summary(self) -> str:
        """Get a summary of the current context."""
        summary = []
        
        if self.medical_context["symptoms"]:
            symptoms = ", ".join(s["text"] for s in self.medical_context["symptoms"])
            summary.append(f"Symptoms: {symptoms}")
        
        if self.medical_context["conditions"]:
            conditions = ", ".join(c["text"] for c in self.medical_context["conditions"])
            summary.append(f"Conditions: {conditions}")
        
        if self.emotional_context["emotions"]:
            emotions = ", ".join(self.emotional_context["emotions"])
            summary.append(f"Emotions: {emotions}")
        
        return " | ".join(summary) if summary else "No specific context"


def generate_response(medical_context: Dict, emotional_context: Dict, empathy_level: str, conversation_state: ConversationState) -> str:
    """Generate a response based on context and empathy level."""
    # Extract key information
    symptoms = [item["text"] for item in medical_context["symptoms"]]
    conditions = [item["text"] for item in medical_context["conditions"]]
    emotions = emotional_context["emotions"]
    
    # Base response templates
    templates = {
        "high": [
            "I understand how challenging this must be for you. {symptom_acknowledgment} Let's work together to find the best way to help you manage this.",
            "I hear the difficulty you're experiencing with {symptoms}. It's completely natural to feel {emotions} about this. Would you like to discuss some ways to help manage these symptoms?",
            "I can see how {symptoms} is affecting you, and I want you to know that I'm here to help. Let's explore some options together to improve your situation."
        ],
        "medium": [
            "I understand that {symptoms} can be concerning. Let's discuss what might be causing this and how we can help.",
            "Thank you for sharing about your {symptoms}. I'd like to understand more about how this is affecting you.",
            "I hear you're dealing with {symptoms}. Would you like to talk about some ways to manage this?"
        ],
        "low": [
            "I see you're experiencing {symptoms}. What would you like to know about this?",
            "Regarding your {symptoms}, could you tell me more about when this started?",
            "I understand you have {symptoms}. Let's discuss this further."
        ]
    }
    
    # Determine empathy level based on context
    if emotions and "negative" in emotions:
        empathy_level = "high"
    elif "severe" in " ".join(symptoms).lower():
        empathy_level = "high"
    elif not symptoms and not conditions:
        empathy_level = "low"
    elif conversation_state.history and len(conversation_state.history) > 2:
        # Increase empathy for persistent issues
        empathy_level = "high"
    
    # Randomly select a template from the appropriate level
    template = random.choice(templates[empathy_level])
    
    # Format response
    symptom_text = ", ".join(symptoms) if symptoms else "your symptoms"
    emotion_text = ", ".join(emotions) if emotions else "concerned"
    
    # Add more context to the response
    if not symptoms and not conditions:
        template = "I understand you're not feeling well. Could you tell me more about your symptoms so I can better help you?"
    elif "severe" in " ".join(symptoms).lower():
        template = "I understand you're experiencing severe symptoms. This must be very difficult for you. Let's discuss this in detail to find the best way to help."
    
    response = template.format(
        symptoms=symptom_text,
        emotions=emotion_text,
        symptom_acknowledgment=f"I can see how {symptom_text} is affecting you" if symptoms else "I understand your situation"
    )
    
    return response


def main():
    # Set device and clear memory
    device = "cpu"  # Force CPU to avoid memory issues
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    print(f"Device set to use {device}")

    try:
        # Initialize components
        print("Initializing components...")
        context_analyzer = ContextAnalyzer(medical_model_name="models/medical_ner")
        empathy_classifier = EmpathyClassifier(
            model_name="distilbert-base-uncased", device=device
        )
        conversation_state = ConversationState()

        print("\nWelcome to the Medical Chat Assistant!")
        print("You can describe your symptoms, concerns, or ask questions.")
        print("Type 'quit' or 'exit' to end the conversation.")
        print("-" * 50)

        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['quit', 'exit']:
                print("\nThank you for chatting. Take care!")
                break
            
            # Analyze context
            medical_context, emotional_context = context_analyzer.analyze_context(user_input)
            
            # Update conversation state
            conversation_state.update_context(medical_context, emotional_context)
            
            # Generate response
            response = generate_response(medical_context, emotional_context, "medium", conversation_state)
            
            # Classify empathy of the generated response
            empathy_level, confidence = empathy_classifier.predict(response)
            
            # Add to history
            conversation_state.add_to_history(user_input, response)
            
            # Print analysis and response
            print("\nAnalysis:")
            print(f"Medical Context: {medical_context}")
            print(f"Emotional Context: {emotional_context}")
            print(f"Response Empathy: {empathy_level} (Confidence: {confidence:.2f})")
            print(f"Conversation Context: {conversation_state.get_context_summary()}")
            print(f"\nAssistant: {response}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        print("\nCleaning up...")
        try:
            del context_analyzer
            del empathy_classifier
            gc.collect()
            print("Cleanup complete")
        except Exception as e:
            print(f"Cleanup error: {str(e)}")
        print("-" * 50)


if __name__ == "__main__":
    main() 