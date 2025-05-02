"""
Open-source medical knowledge base integration.
"""

from typing import Dict, List
import json
import os
from pathlib import Path


class MedicalKnowledgeBase:
    """Open-source medical knowledge base integration."""

    def __init__(self, data_dir: str = "data/medical"):
        self.data_dir = Path(data_dir)
        self.cache = {}
        self.medical_data = {
            "conditions": {},
            "symptoms": {},
            "treatments": {},
            "medications": {},
            "relationships": {},
        }

        # Load medical resources
        self.load_resources()

    def load_resources(self):
        """Load medical resources from open-source databases."""
        try:
            # Ensure data directory exists
            os.makedirs(self.data_dir, exist_ok=True)

            # Load each resource type
            for resource_type in self.medical_data.keys():
                resource_file = self.data_dir / f"{resource_type}.json"
                if resource_file.exists():
                    with open(resource_file, "r") as f:
                        self.medical_data[resource_type] = json.load(f)
        except Exception as e:
            print(f"Error loading medical resources: {str(e)}")

    def query_condition(self, condition: str) -> Dict:
        """Query condition information."""
        if condition in self.cache:
            return self.cache[condition]

        # Query condition information
        info = {
            "name": condition,
            "symptoms": [],
            "treatments": [],
            "medications": [],
            "severity": "unknown",
            "common_complications": [],
            "references": [],
        }

        # Look up in medical data
        if condition.lower() in self.medical_data["conditions"]:
            condition_data = self.medical_data["conditions"][condition.lower()]
            info.update(condition_data)

            # Get related symptoms
            if "symptom_refs" in condition_data:
                for symptom_ref in condition_data["symptom_refs"]:
                    if symptom_ref in self.medical_data["symptoms"]:
                        info["symptoms"].append(
                            self.medical_data["symptoms"][symptom_ref]
                        )

            # Get related treatments
            if "treatment_refs" in condition_data:
                for treatment_ref in condition_data["treatment_refs"]:
                    if treatment_ref in self.medical_data["treatments"]:
                        info["treatments"].append(
                            self.medical_data["treatments"][treatment_ref]
                        )

        # Cache results
        self.cache[condition] = info
        return info

    def validate_medical_terms(self, terms: List[str]) -> Dict:
        """Validate medical terms against knowledge base."""
        validated = {}
        for term in terms:
            term_lower = term.lower()
            validation = {
                "valid": False,
                "normalized_term": term,
                "category": "unknown",
                "confidence": 0.0,
            }

            # Check each category
            for category, data in self.medical_data.items():
                if term_lower in data:
                    validation.update(
                        {
                            "valid": True,
                            "normalized_term": data[term_lower].get(
                                "preferred_name", term
                            ),
                            "category": category,
                            "confidence": 1.0,
                        }
                    )
                    break

            validated[term] = validation

        return validated

    def get_related_terms(self, term: str, category: str = None) -> List[Dict]:
        """Get related medical terms."""
        related = []
        term_lower = term.lower()

        if category and category in self.medical_data:
            # Search in specific category
            if term_lower in self.medical_data[category]:
                related_refs = self.medical_data[category][term_lower].get(
                    "related_refs", []
                )
                for ref in related_refs:
                    if ref in self.medical_data[category]:
                        related.append(self.medical_data[category][ref])
        else:
            # Search across all categories
            for category_data in self.medical_data.values():
                if term_lower in category_data:
                    related_refs = category_data[term_lower].get("related_refs", [])
                    for ref in related_refs:
                        if ref in category_data:
                            related.append(category_data[ref])

        return related

    def add_medical_data(self, category: str, data: Dict):
        """Add new medical data to the knowledge base."""
        if category in self.medical_data:
            self.medical_data[category].update(data)

            # Save to file
            resource_file = self.data_dir / f"{category}.json"
            with open(resource_file, "w") as f:
                json.dump(self.medical_data[category], f, indent=2)

    def get_categories(self) -> List[str]:
        """Get available medical data categories."""
        return list(self.medical_data.keys())
