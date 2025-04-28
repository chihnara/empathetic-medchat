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
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def print_separator(message=""):
    """Print a separator line with optional message."""
    width = 80
    if message:
        message = f" {message} "
        padding = (width - len(message)) // 2
        print("\n" + "=" * padding + message + "=" * padding)
    else:
        print("\n" + "=" * width)


def main():
    """Main function to run the MEDICOD system."""
    print_separator("Initializing MEDICOD System")

    # Set device
    device = "cpu"
    print(f"\nDevice: {device.upper()}")

    print("\nInitializing components...")

    try:
        # Initialize context analyzer
        print("Testing context analyzer...")
        context_analyzer = ContextAnalyzer()
        logger.info("Context analyzer initialized successfully")

        # Initialize empathy classifier
        print("Testing empathy classifier...")
        empathy_classifier = EmpathyClassifier()
        logger.info("Empathy classifier initialized successfully")

        # Initialize response generator
        print("Testing response generator...")
        response_generator = ResponseGenerator()
        logger.info("Response generator initialized successfully")

        # Test queries
        test_queries = [
            {
                "query": "I've been having severe headaches and feeling dizzy lately. It's really affecting my daily life.",
                "expected_symptoms": ["severe headaches", "dizziness"],
                "expected_emotions": ["worried", "distressed"],
            },
            {
                "query": "My chronic back pain is getting worse and I'm feeling hopeless about finding relief.",
                "expected_symptoms": ["chronic back pain"],
                "expected_emotions": ["hopeless", "depressed"],
            },
            {
                "query": "I'm concerned about my high blood pressure and diabetes. The medications don't seem to be helping.",
                "expected_symptoms": ["high blood pressure", "diabetes"],
                "expected_emotions": ["concerned", "anxious"],
            },
        ]

        print("\nProcessing test queries...")
        for query_data in test_queries:
            query = query_data["query"]
            print(f"\nQuery: {query}")

            # Analyze context
            print("\nAnalyzing context...")
            medical_context, emotional_context = context_analyzer.analyze_context(query)
            print(f"Medical context: {medical_context}")
            print(f"Emotional context: {emotional_context}")

            # Generate responses with different empathy levels and find the best one
            print("\nGenerating best response...")
            best_response = None
            best_confidence = 0.0
            best_level = None

            for empathy_level in ["high", "medium", "low"]:
                response = response_generator.generate_response(
                    query, medical_context, emotional_context, empathy_level
                )
                predicted_level, confidence = empathy_classifier.predict(response)

                if confidence > best_confidence:
                    best_response = response
                    best_confidence = confidence
                    best_level = predicted_level

            print(
                f"\nBest response ({best_level} empathy, confidence: {best_confidence:.2f}):"
            )
            print(best_response)

    except Exception as e:
        logger.error(f"\nError occurred: {str(e)}")
        raise

    finally:
        print_separator("Cleaning Up")
        try:
            if "empathy_classifier" in locals():
                del empathy_classifier
            if "context_analyzer" in locals():
                del context_analyzer
            if "response_generator" in locals():
                del response_generator
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")


if __name__ == "__main__":
    main()
