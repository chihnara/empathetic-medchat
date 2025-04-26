"""
Script to prepare training data for medical NER model.
"""

import json
from typing import Dict, List, Tuple
import spacy
from spacy.tokens import Doc
from spacy.training import Example
import random


def load_medical_dialogue_data(file_path: str) -> List[Dict]:
    """Load medical dialogue data from file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def annotate_medical_entities(
    text: str,
    nlp: spacy.Language,
    entity_types: Dict[str, str] = {
        "SYMPTOM": "symptoms",
        "DISEASE": "conditions",
        "TREATMENT": "treatments",
        "DRUG": "medications",
    },
) -> List[str]:
    """Annotate medical entities in text using spaCy."""
    doc = nlp(text)
    tokens = [token.text for token in doc]
    labels = ["O"] * len(tokens)

    for ent in doc.ents:
        if ent.label_ in entity_types:
            # Mark beginning of entity
            labels[ent.start] = f"B-{ent.label_}"
            # Mark continuation of entity
            for i in range(ent.start + 1, ent.end):
                labels[i] = f"I-{ent.label_}"

    return labels


def prepare_training_data(
    dialogue_data: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Prepare training and validation data from dialogue data."""
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Add medical entity patterns
    patterns = [
        {
            "label": "SYMPTOM",
            "pattern": [{"LOWER": {"IN": ["pain", "ache", "discomfort"]}}],
        },
        {
            "label": "SYMPTOM",
            "pattern": [{"LOWER": {"IN": ["headache", "dizziness", "nausea"]}}],
        },
        {
            "label": "CONDITION",
            "pattern": [{"LOWER": {"IN": ["diabetes", "hypertension", "pressure"]}}],
        },
        {
            "label": "TREATMENT",
            "pattern": [{"LOWER": {"IN": ["medication", "treatment", "therapy"]}}],
        },
        {
            "label": "DRUG",
            "pattern": [{"LOWER": {"IN": ["aspirin", "ibuprofen", "paracetamol"]}}],
        },
    ]
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)

    # Prepare data
    prepared_data = []
    for dialogue in dialogue_data:
        text = dialogue.get("text", "")
        if text:
            labels = annotate_medical_entities(text, nlp)
            prepared_data.append({"text": text, "labels": labels})

    # Split into train and validation sets
    random.seed(seed)
    random.shuffle(prepared_data)
    split_idx = int(len(prepared_data) * train_ratio)
    train_data = prepared_data[:split_idx]
    val_data = prepared_data[split_idx:]

    return train_data, val_data


def save_data(data: List[Dict], file_path: str) -> None:
    """Save prepared data to file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    # Example usage
    dialogue_data = [
        {"text": "I have a severe headache and dizziness"},
        {"text": "My chronic back pain is getting worse"},
        {"text": "I've been diagnosed with high blood pressure"},
        {"text": "The doctor prescribed me some pain medication"},
    ]

    train_data, val_data = prepare_training_data(dialogue_data)

    # Save prepared data
    save_data(train_data, "data/train_data.json")
    save_data(val_data, "data/val_data.json")
