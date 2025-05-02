"""
Enhanced paraphrasing module with modern LLM integration.
"""

from typing import Dict, List
import random


class EnhancedParaphraser:
    """Enhanced paraphrasing with modern LLM integration."""

    def __init__(self):
        self.medical_terms = set()  # Load from medical knowledge base
        self.response_templates = {
            "concern": [
                "I understand your concern about {symptom}. {response}",
                "I can see why {symptom} is worrying you. {response}",
                "Let's address your concerns about {symptom}. {response}",
            ],
            "reassurance": [
                "While {symptom} can be concerning, {response}",
                "Many people experience {symptom}, and {response}",
                "Although {symptom} is troubling, {response}",
            ],
            "instruction": [
                "To manage {symptom}, you should {response}",
                "Here's what you can do about {symptom}: {response}",
                "For {symptom}, I recommend {response}",
            ],
            "empathy": [
                "I hear how {symptom} is affecting you. {response}",
                "Living with {symptom} can be challenging. {response}",
                "I understand the impact {symptom} has on you. {response}",
            ],
            "encouragement": [
                "You're taking the right steps to address {symptom}. {response}",
                "It's great that you're seeking help for {symptom}. {response}",
                "Together we can work on managing {symptom}. {response}",
            ],
        }

    def paraphrase(self, text: str, context: Dict, style: str = "concern") -> str:
        """Generate contextually appropriate paraphrases."""
        try:
            # Extract medical terms
            medical_terms = self._extract_medical_terms(text)

            # Select appropriate template
            templates = self.response_templates.get(
                style, self.response_templates["concern"]
            )
            template = random.choice(templates)

            # Format response
            symptom = context.get("symptoms", [{"text": "your symptoms"}])[0]["text"]
            response = text.replace(symptom, "").strip()

            return template.format(symptom=symptom, response=response)
        except Exception as e:
            print(f"Error in paraphrasing: {str(e)}")
            return text  # Return original text if paraphrasing fails

    def _extract_medical_terms(self, text: str) -> List[str]:
        """Extract medical terms from text."""
        terms = []
        for word in text.split():
            if word.lower() in self.medical_terms:
                terms.append(word)
        return terms

    def add_medical_terms(self, terms: List[str]):
        """Add medical terms to the knowledge base."""
        self.medical_terms.update(terms)

    def get_template_styles(self) -> List[str]:
        """Get available template styles."""
        return list(self.response_templates.keys())
