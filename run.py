"""
End-to-end testing of MEDICOD implementation.
"""

import os
import torch
import gc
from typing import Dict, List, Tuple
from src.models.context_analyzer import ContextAnalyzer
from src.models.empathy_classifier import EmpathyClassifier
import time


def main():
    # Set device and clear memory
    device = "cpu"  # Force CPU to avoid memory issues
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    print(f"Device set to use {device}")

    try:
        # Initialize components with minimal models
        print("Initializing components...")

        # Initialize context analyzer
        print("Testing context analyzer...")
        context_analyzer = ContextAnalyzer(medical_model_name="models/medical_ner")

        # Initialize empathy classifier
        print("Testing empathy classifier...")
        empathy_classifier = EmpathyClassifier(
            model_name="distilbert-base-uncased", device=device
        )

        # Test queries and responses
        test_cases = [
            {
                "query": "I've been having severe headaches and dizziness for the past week. I'm really worried about what this could be.",
                "response": "I understand that experiencing severe headaches and dizziness can be very concerning. It's natural to feel worried about these symptoms. Let's work together to understand what might be causing them and determine the best course of action.",
            },
            {
                "query": "My chronic back pain is getting worse, and I can barely sleep at night. I feel hopeless.",
                "response": "I hear how much the worsening back pain is affecting you, especially with your sleep being disrupted. It must be incredibly frustrating and discouraging to deal with this chronic pain. Let's discuss some ways to help manage your pain and improve your sleep.",
            },
            {
                "query": "I've been diagnosed with high blood pressure and diabetes. I don't know how to handle all these medications.",
                "response": "Managing multiple conditions like high blood pressure and diabetes can feel overwhelming. I can see why keeping track of different medications might be challenging. Would you like to go through your medication schedule together to make it more manageable?",
            },
        ]

        print("\nRunning test cases...")
        print("-" * 50)

        for case in test_cases:
            print(f"\nProcessing query: {case['query']}")

            # Analyze context
            medical_context, emotional_context = context_analyzer.analyze_context(
                case["query"]
            )
            print("\nContext Analysis:")
            print(f"Medical Context: {medical_context}")
            print(f"Emotional Context: {emotional_context}")

            # Classify empathy
            empathy_level, confidence = empathy_classifier.predict(case["response"])
            print("\nEmpathy Classification:")
            print(f"Response: {case['response']}")
            print(f"Empathy Level: {empathy_level} (Confidence: {confidence:.2f})")
            print("-" * 50)

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
