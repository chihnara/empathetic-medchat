"""
Empathy classification for medical dialogue using rule-based approach.
"""

from typing import Dict, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class EmpathyClassifier:
    """Classifies the level of empathy in medical responses using a rule-based approach."""

    def __init__(self):
        """Initialize the empathy classifier with pattern matching rules."""
        self.logger = logging.getLogger(__name__)

        # Enhanced empathy indicators with weights
        self.empathy_indicators = {
            "high": {
                "acknowledgment": [
                    (r"i\s+understand", 1.0),
                    (r"i\s+hear\s+your", 1.0),
                    (r"feelings?\s+are\s+valid", 1.0),
                    (r"must\s+be\s+difficult", 0.9),
                    (r"sounds?\s+challenging", 0.9),
                ],
                "support": [
                    (r"here\s+to\s+support\s+you", 1.0),
                    (r"get\s+through\s+this\s+together", 1.0),
                    (r"not\s+alone", 0.9),
                    (r"be\s+with\s+you", 0.9),
                ],
                "personalization": [
                    (r"works?\s+best\s+for\s+you", 1.0),
                    (r"tailor\s+this", 0.9),
                    (r"individual\s+circumstances?", 0.8),
                ],
            },
            "medium": {
                "acknowledgment": [
                    (r"i\s+see\s+you", 0.8),
                    (r"sounds?\s+concerning", 0.8),
                    (r"must\s+be\s+hard", 0.7),
                ],
                "support": [
                    (r"work\s+on\s+this\s+together", 0.8),
                    (r"let'?s\s+address", 0.7),
                    (r"help\s+you\s+with", 0.7),
                ],
                "personalization": [
                    (r"find\s+a\s+solution", 0.8),
                    (r"explore\s+your\s+options", 0.7),
                ],
            },
            "low": {
                "acknowledgment": [
                    (r"look\s+at\s+your", 0.6),
                    (r"address\s+this", 0.6),
                    (r"your\s+symptoms", 0.5),
                ],
                "support": [
                    (r"what\s+we\s+can\s+do", 0.6),
                    (r"focus\s+on\s+treatment", 0.5),
                ],
                "personalization": [
                    (r"standard\s+treatment", 0.5),
                    (r"evidence-based", 0.5),
                ],
            },
        }

        # Sentiment indicators
        self.sentiment_indicators = {
            "positive": [
                r"understand",
                r"support",
                r"help",
                r"together",
                r"guide",
                r"care",
                r"better",
            ],
            "negative": [
                r"just",
                r"only",
                r"must",
                r"should",
                r"need to",
                r"have to",
            ],
        }

        # Response characteristics weights
        self.weights = {
            "indicators": 0.4,  # Weight for empathy indicators
            "sentiment": 0.2,  # Weight for sentiment words
            "pronouns": 0.2,  # Weight for personal pronouns
            "length": 0.1,  # Weight for response length
            "questions": 0.1,  # Weight for questions
        }

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment using keyword matching."""
        text = text.lower()
        positive_count = sum(
            len(re.findall(pattern, text))
            for pattern in self.sentiment_indicators["positive"]
        )
        negative_count = sum(
            len(re.findall(pattern, text))
            for pattern in self.sentiment_indicators["negative"]
        )

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        return (positive_count - negative_count) / total

    def _count_empathy_indicators(self, text: str) -> Dict[str, float]:
        """Count empathy indicators in the text with weighted scoring."""
        text = text.lower()
        scores = {"high": 0.0, "medium": 0.0, "low": 0.0}

        for level, categories in self.empathy_indicators.items():
            for category, patterns in categories.items():
                for pattern, weight in patterns:
                    matches = len(re.findall(pattern, text))
                    scores[level] += matches * weight

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores

    def _analyze_characteristics(self, text: str) -> Dict[str, float]:
        """Analyze various characteristics of the response."""
        text = text.lower()
        words = text.split()

        characteristics = {
            "length": min(len(words) / 50.0, 1.0),  # Normalize by expected length
            "questions": min(
                text.count("?") / 2.0, 1.0
            ),  # Normalize by expected questions
            "pronouns": min(
                len(re.findall(r"\b(i|you|we|your|our)\b", text)) / 5.0, 1.0
            ),  # Normalize by expected pronouns
        }

        return characteristics

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict the empathy level of a response.

        Args:
            text: The response text to analyze

        Returns:
            Tuple of (empathy_level, confidence_score)
        """
        try:
            # Get base scores
            indicator_scores = self._count_empathy_indicators(text)
            sentiment_score = self._analyze_sentiment(text)
            characteristics = self._analyze_characteristics(text)

            # Calculate final scores
            final_scores = {}
            for level in ["high", "medium", "low"]:
                # Start with indicator score
                score = indicator_scores[level] * self.weights["indicators"]

                # Add sentiment contribution
                if level == "high":
                    score += max(0, sentiment_score) * self.weights["sentiment"]
                elif level == "low":
                    score += max(0, -sentiment_score) * self.weights["sentiment"]

                # Add characteristics
                score += (
                    characteristics["length"] * self.weights["length"]
                    + characteristics["questions"] * self.weights["questions"]
                    + characteristics["pronouns"] * self.weights["pronouns"]
                )

                final_scores[level] = score

            # Get highest scoring level
            max_level = max(final_scores.items(), key=lambda x: x[1])
            level, confidence = max_level

            # Normalize confidence to 0-1 range
            total_score = sum(final_scores.values())
            if total_score > 0:
                confidence = confidence / total_score
            else:
                confidence = 0.5  # Default confidence if no scores
                level = "medium"  # Default to medium if no clear indicators

            return level, min(confidence, 0.95)  # Cap confidence at 0.95

        except Exception as e:
            self.logger.error(f"Error in empathy prediction: {str(e)}")
            return "medium", 0.5  # Default to medium on error
