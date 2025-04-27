"""
End-to-end testing of MEDICOD implementation.
"""

import os
import torch
import gc
from typing import Dict, List, Tuple
from src.models.context_analyzer import ContextAnalyzer
from src.models.empathy_classifier import EmpathyClassifier
from src.models.response_generator import ResponseGenerator
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
        context_analyzer = ContextAnalyzer(medical_model_name="src/models/medical_ner")

        # Initialize empathy classifier
        print("Testing empathy classifier...")
        empathy_classifier = EmpathyClassifier(
            model_name="distilbert-base-uncased", device=device
        )

        # Initialize response generator
        print("Testing response generator...")
        response_generator = ResponseGenerator()

        # Test queries
        test_queries = [
            "I've been having severe headaches and dizziness for the past week. I'm really worried about what this could be.",
            "My chronic back pain is getting worse, and I can barely sleep at night. I feel hopeless.",
            "I've been diagnosed with high blood pressure and diabetes. I don't know how to handle all these medications.",
        ]

        print("\nRunning test cases...")
        print("-" * 50)

        for query in test_queries:
            print(f"\nProcessing query: {query}")

            # Analyze context
            medical_context, emotional_context = context_analyzer.analyze_context(query)
            print("\nContext Analysis:")
            print(f"Medical Context: {medical_context}")
            print(f"Emotional Context: {emotional_context}")

            # Generate response
            context = {"medical": medical_context, "emotional": emotional_context}
            response = response_generator.generate_response(
                context, empathy_level="high"
            )
            print("\nGenerated Response:")
            print(response)

            # Classify empathy of generated response
            empathy_level, confidence = empathy_classifier.predict(response)
            print("\nEmpathy Classification:")
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
            del response_generator
            gc.collect()
            print("Cleanup complete")
        except Exception as e:
            print(f"Cleanup error: {str(e)}")
        print("-" * 50)


if __name__ == "__main__":
    main()
