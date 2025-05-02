import json
import sys
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tabulate import tabulate
import time
import psutil
import os

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)


class MEDCODEvaluator:
    def __init__(self):
        self.base_url = "http://localhost:5000"

        # Load test cases from JSON file
        test_cases_path = Path(__file__).parent / "test_cases.json"
        with open(test_cases_path, "r") as f:
            test_data = json.load(f)
            self.test_cases = test_data["test_cases"]

    def evaluate_medical_accuracy(
        self, responses: List[Dict[str, Any]], test_cases: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate medical accuracy with detailed metrics."""
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        category_metrics = {}

        for response, test_case in zip(responses, test_cases):
            # Extract symptoms from response, handling both string and dict formats
            detected_symptoms_raw = response.get("medical_context", {}).get(
                "symptoms", []
            )
            detected_symptoms = []
            for symptom in detected_symptoms_raw:
                if isinstance(symptom, dict):
                    detected_symptoms.append(symptom.get("text", "").lower())
                else:
                    detected_symptoms.append(str(symptom).lower())

            # Convert expected symptoms to lowercase for comparison
            expected_symptoms = [s.lower() for s in test_case["expected_symptoms"]]
            category = test_case["category"]

            if category not in category_metrics:
                category_metrics[category] = {
                    "precision": [],
                    "recall": [],
                    "f1": [],
                    "found": [],
                    "expected": [],
                }

            # Calculate metrics using text comparison
            true_positives = len(set(detected_symptoms) & set(expected_symptoms))
            false_positives = len(set(detected_symptoms) - set(expected_symptoms))
            false_negatives = len(set(expected_symptoms) - set(detected_symptoms))

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            category_metrics[category]["precision"].append(precision)
            category_metrics[category]["recall"].append(recall)
            category_metrics[category]["f1"].append(f1)
            category_metrics[category]["found"].append(len(detected_symptoms))
            category_metrics[category]["expected"].append(len(expected_symptoms))

            total_precision += precision
            total_recall += recall
            total_f1 += f1

        avg_precision = total_precision / len(responses)
        avg_recall = total_recall / len(responses)
        avg_f1 = total_f1 / len(responses)

        return avg_f1, {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "category_metrics": category_metrics,
        }

    def evaluate_emotional_quality(
        self, responses: List[Dict[str, Any]], test_cases: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate emotional quality with detailed metrics."""
        total_score = 0
        category_metrics = {}

        # Define emotion similarity groups
        emotion_groups = {
            "anxiety": ["anxiety", "worry", "nervousness", "panic"],
            "fear": ["fear", "scared", "terrified", "dread"],
            "concern": ["concern", "worried", "apprehensive"],
            "depression": ["depression", "sadness", "down", "low"],
            "frustration": ["frustration", "annoyed", "irritated"],
            "guilt": ["guilt", "burden", "shame"],
            "hopelessness": ["hopeless", "despair", "helpless"],
            "loneliness": ["lonely", "isolated", "alone"],
            "pride": ["pride", "accomplished", "achievement"],
            "gratitude": ["gratitude", "thankful", "grateful"],
            "mixed": ["mixed", "conflicted", "uncertain"],
            "urgent": ["urgent", "emergency", "critical"],
        }

        for response, test_case in zip(responses, test_cases):
            detected_emotions = [
                e.lower()
                for e in response.get("emotional_context", {}).get("emotions", [])
            ]
            expected_emotion = test_case["expected_emotion"].lower()
            category = test_case["category"]

            if category not in category_metrics:
                category_metrics[category] = {
                    "empathy_levels": [],
                    "emotion_matches": [],
                    "empathy_scores": [],
                    "confidences": [],
                    "response_appropriateness": [],
                }

            # Calculate emotion match with similarity groups
            emotion_match = 0
            if expected_emotion in detected_emotions:
                emotion_match = 1
            else:
                # Check if detected emotions are in the same group as expected emotion
                for group in emotion_groups.values():
                    if expected_emotion in group:
                        if any(e in group for e in detected_emotions):
                            emotion_match = 0.75
                            break

            # Calculate empathy level score
            empathy_level = response.get("emotional_context", {}).get(
                "empathy_level", "low"
            )
            empathy_scores = {"low": 0.3, "medium": 0.6, "high": 1.0}
            empathy_score = empathy_scores.get(empathy_level, 0.3)

            # Calculate response appropriateness based on category
            response_text = response.get("response", "").lower()
            appropriateness_score = 0.0

            # Check for validation phrases
            validation_phrases = [
                "understand",
                "hear",
                "must be",
                "sounds",
                "appreciate",
                "that's difficult",
                "challenging",
                "not easy",
            ]
            if any(phrase in response_text for phrase in validation_phrases):
                appropriateness_score += 0.5

            # Check for empathetic acknowledgment
            acknowledgment_phrases = [
                "i'm sorry",
                "it's understandable",
                "it makes sense",
                "that's frustrating",
                "that's scary",
            ]
            if any(phrase in response_text for phrase in acknowledgment_phrases):
                appropriateness_score += 0.5

            # Get confidence score
            confidence = response.get("emotional_context", {}).get(
                "overall_confidence", 0.5
            )

            # Calculate combined score with weights
            combined_score = (
                emotion_match * 0.4  # Emotion recognition
                + empathy_score * 0.3  # Empathy level
                + appropriateness_score * 0.2  # Response appropriateness
                + confidence * 0.1  # Confidence
            )

            category_metrics[category]["empathy_levels"].append(empathy_level)
            category_metrics[category]["emotion_matches"].append(emotion_match)
            category_metrics[category]["empathy_scores"].append(empathy_score)
            category_metrics[category]["confidences"].append(confidence)
            category_metrics[category]["response_appropriateness"].append(
                appropriateness_score
            )

            total_score += combined_score

        avg_score = total_score / len(responses)

        return avg_score, {"score": avg_score, "category_metrics": category_metrics}

    def evaluate_response_diversity(
        self, responses: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate response diversity with detailed metrics."""
        unique_responses = set()
        total_length = 0

        for response in responses:
            response_text = response.get("response", "")
            unique_responses.add(response_text)
            total_length += len(response_text.split())

        unique_ratio = len(unique_responses) / len(responses)
        avg_length = total_length / len(responses)

        return unique_ratio, {
            "unique_ratio": unique_ratio,
            "unique_responses": len(unique_responses),
            "total_responses": len(responses),
            "avg_length": avg_length,
        }

    def evaluate_controllability(
        self, responses: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate controllability with detailed metrics."""
        empathy_levels = []
        context_consistency = 0

        for response in responses:
            empathy_level = response.get("emotional_context", {}).get(
                "empathy_level", "low"
            )
            empathy_levels.append(empathy_level)

            # Check if response maintains context
            if response.get("medical_context") and response.get("emotional_context"):
                context_consistency += 1

        # Calculate empathy level consistency
        unique_empathy_levels = len(set(empathy_levels))
        empathy_consistency = 1 - (unique_empathy_levels / len(empathy_levels))

        # Calculate context consistency
        context_consistency = context_consistency / len(responses)

        # Overall controllability score
        controllability_score = (empathy_consistency + context_consistency) / 2

        return controllability_score, {
            "empathy_consistency": empathy_consistency,
            "context_consistency": context_consistency,
            "unique_empathy_levels": unique_empathy_levels,
            "total_responses": len(responses),
        }

    def evaluate_deployment_metrics(
        self, responses: List[Dict[str, Any]], response_times: List[float]
    ) -> Dict[str, Any]:
        """Evaluate practical deployment metrics."""
        # Calculate response time metrics
        avg_response_time = np.mean(response_times)
        max_response_time = np.max(response_times)
        min_response_time = np.min(response_times)

        # Calculate memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB

        # Calculate error rate
        error_count = sum(1 for r in responses if "error" in r)
        error_rate = error_count / len(responses)

        return {
            "avg_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "min_response_time": min_response_time,
            "memory_usage_mb": memory_usage,
            "error_rate": error_rate,
            "total_requests": len(responses),
        }

    def evaluate_version(self, version: str) -> Dict[str, Any]:
        """Evaluate a specific version of the system."""
        responses = []
        response_times = []

        for test_case in self.test_cases:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/{version}/chat", json={"message": test_case["input"]}
            )
            end_time = time.time()

            response_times.append(end_time - start_time)
            responses.append(response.json())

        # Evaluate all metrics
        medical_accuracy, medical_details = self.evaluate_medical_accuracy(
            responses, self.test_cases
        )
        emotional_quality, emotional_details = self.evaluate_emotional_quality(
            responses, self.test_cases
        )
        response_diversity, diversity_details = self.evaluate_response_diversity(
            responses
        )
        controllability, control_details = self.evaluate_controllability(responses)
        deployment_metrics = self.evaluate_deployment_metrics(responses, response_times)

        return {
            "medical_accuracy": medical_accuracy,
            "emotional_quality": emotional_quality,
            "response_diversity": response_diversity,
            "controllability": controllability,
            "medical_details": medical_details,
            "emotional_details": emotional_details,
            "diversity_details": diversity_details,
            "control_details": control_details,
            "deployment_metrics": deployment_metrics,
        }

    def print_detailed_results(
        self, v1_results: Dict[str, Any], v2_results: Dict[str, Any]
    ):
        """Print detailed evaluation results with v1 vs v2 comparison."""
        print("\nDetailed Evaluation Results\n")

        # Overall scores comparison
        print("1. Overall Scores Comparison:")
        headers = ["Metric", "V1 Score", "V2 Score", "Improvement", "% Change"]
        data = [
            [
                "Medical Accuracy",
                v1_results["medical_accuracy"],
                v2_results["medical_accuracy"],
                v2_results["medical_accuracy"] - v1_results["medical_accuracy"],
                (
                    (
                        (
                            v2_results["medical_accuracy"]
                            - v1_results["medical_accuracy"]
                        )
                        / v1_results["medical_accuracy"]
                        * 100
                    )
                    if v1_results["medical_accuracy"] != 0
                    else 0
                ),
            ],
            [
                "Emotional Quality",
                v1_results["emotional_quality"],
                v2_results["emotional_quality"],
                v2_results["emotional_quality"] - v1_results["emotional_quality"],
                (
                    (
                        (
                            v2_results["emotional_quality"]
                            - v1_results["emotional_quality"]
                        )
                        / v1_results["emotional_quality"]
                        * 100
                    )
                    if v1_results["emotional_quality"] != 0
                    else 0
                ),
            ],
            [
                "Response Diversity",
                v1_results["response_diversity"],
                v2_results["response_diversity"],
                v2_results["response_diversity"] - v1_results["response_diversity"],
                (
                    (
                        (
                            v2_results["response_diversity"]
                            - v1_results["response_diversity"]
                        )
                        / v1_results["response_diversity"]
                        * 100
                    )
                    if v1_results["response_diversity"] != 0
                    else 0
                ),
            ],
            [
                "Controllability",
                v1_results["controllability"],
                v2_results["controllability"],
                v2_results["controllability"] - v1_results["controllability"],
                (
                    (
                        (v2_results["controllability"] - v1_results["controllability"])
                        / v1_results["controllability"]
                        * 100
                    )
                    if v1_results["controllability"] != 0
                    else 0
                ),
            ],
        ]
        print(tabulate(data, headers=headers, tablefmt="grid"))

        # Medical accuracy details by category
        print("\n2. Medical Accuracy by Category:")
        headers = ["Category", "V1 F1", "V2 F1", "Improvement", "% Change"]
        data = []
        for category in v1_results["medical_details"]["category_metrics"]:
            v1_f1 = np.mean(
                v1_results["medical_details"]["category_metrics"][category]["f1"]
            )
            v2_f1 = np.mean(
                v2_results["medical_details"]["category_metrics"][category]["f1"]
            )
            improvement = v2_f1 - v1_f1
            percent_change = (improvement / v1_f1 * 100) if v1_f1 != 0 else 0
            data.append([category, v1_f1, v2_f1, improvement, percent_change])
        print(tabulate(data, headers=headers, tablefmt="grid"))

        # Emotional quality details
        print("\n3. Emotional Quality by Category:")
        headers = ["Category", "V1 Score", "V2 Score", "Improvement", "% Change"]
        data = []
        for category in v1_results["emotional_details"]["category_metrics"]:
            v1_score = np.mean(
                v1_results["emotional_details"]["category_metrics"][category][
                    "empathy_scores"
                ]
            )
            v2_score = np.mean(
                v2_results["emotional_details"]["category_metrics"][category][
                    "empathy_scores"
                ]
            )
            improvement = v2_score - v1_score
            percent_change = (improvement / v1_score * 100) if v1_score != 0 else 0
            data.append([category, v1_score, v2_score, improvement, percent_change])
        print(tabulate(data, headers=headers, tablefmt="grid"))

        # Deployment metrics
        print("\n4. Deployment Metrics:")
        headers = ["Metric", "V1", "V2", "Improvement", "% Change"]
        data = [
            [
                "Avg Response Time (s)",
                v1_results["deployment_metrics"]["avg_response_time"],
                v2_results["deployment_metrics"]["avg_response_time"],
                v2_results["deployment_metrics"]["avg_response_time"]
                - v1_results["deployment_metrics"]["avg_response_time"],
                (
                    (
                        (
                            v2_results["deployment_metrics"]["avg_response_time"]
                            - v1_results["deployment_metrics"]["avg_response_time"]
                        )
                        / v1_results["deployment_metrics"]["avg_response_time"]
                        * 100
                    )
                    if v1_results["deployment_metrics"]["avg_response_time"] != 0
                    else 0
                ),
            ],
            [
                "Memory Usage (MB)",
                v1_results["deployment_metrics"]["memory_usage_mb"],
                v2_results["deployment_metrics"]["memory_usage_mb"],
                v2_results["deployment_metrics"]["memory_usage_mb"]
                - v1_results["deployment_metrics"]["memory_usage_mb"],
                (
                    (
                        (
                            v2_results["deployment_metrics"]["memory_usage_mb"]
                            - v1_results["deployment_metrics"]["memory_usage_mb"]
                        )
                        / v1_results["deployment_metrics"]["memory_usage_mb"]
                        * 100
                    )
                    if v1_results["deployment_metrics"]["memory_usage_mb"] != 0
                    else 0
                ),
            ],
            [
                "Error Rate",
                v1_results["deployment_metrics"]["error_rate"],
                v2_results["deployment_metrics"]["error_rate"],
                v2_results["deployment_metrics"]["error_rate"]
                - v1_results["deployment_metrics"]["error_rate"],
                (
                    (
                        (
                            v2_results["deployment_metrics"]["error_rate"]
                            - v1_results["deployment_metrics"]["error_rate"]
                        )
                        / v1_results["deployment_metrics"]["error_rate"]
                        * 100
                    )
                    if v1_results["deployment_metrics"]["error_rate"] != 0
                    else 0
                ),
            ],
        ]
        print(tabulate(data, headers=headers, tablefmt="grid"))

    def run_evaluation(self):
        """Run the complete evaluation."""
        print("Starting detailed evaluation...")

        print("\nEvaluating V1 implementation...")
        v1_results = self.evaluate_version("v1")

        print("\nEvaluating V2 implementation...")
        v2_results = self.evaluate_version("v2")

        self.print_detailed_results(v1_results, v2_results)


if __name__ == "__main__":
    evaluator = MEDCODEvaluator()
    evaluator.run_evaluation()
