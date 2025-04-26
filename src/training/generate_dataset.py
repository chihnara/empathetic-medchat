"""
Script to generate a large medical dialogue dataset for training.
"""

import json
import random
from typing import List, Dict
import os

# Medical conditions and their associated symptoms
MEDICAL_CONDITIONS = {
    "hypertension": [
        "high blood pressure",
        "headache",
        "dizziness",
        "chest pain",
        "fatigue",
    ],
    "diabetes": [
        "frequent urination",
        "increased thirst",
        "fatigue",
        "blurred vision",
        "slow healing",
    ],
    "asthma": [
        "shortness of breath",
        "wheezing",
        "coughing",
        "chest tightness",
        "difficulty breathing",
    ],
    "arthritis": [
        "joint pain",
        "stiffness",
        "swelling",
        "reduced range of motion",
        "fatigue",
    ],
    "depression": [
        "sadness",
        "loss of interest",
        "fatigue",
        "sleep problems",
        "appetite changes",
    ],
    "anxiety": [
        "nervousness",
        "restlessness",
        "rapid heartbeat",
        "sweating",
        "trouble concentrating",
    ],
    "migraine": [
        "severe headache",
        "nausea",
        "sensitivity to light",
        "sensitivity to sound",
        "aura",
    ],
    "gastroenteritis": ["diarrhea", "nausea", "vomiting", "abdominal pain", "fever"],
    "pneumonia": ["cough", "fever", "shortness of breath", "chest pain", "fatigue"],
    "allergies": ["sneezing", "runny nose", "itchy eyes", "congestion", "rash"],
}

# Common medications and treatments
MEDICATIONS = {
    "pain": ["ibuprofen", "acetaminophen", "aspirin", "naproxen"],
    "allergy": ["antihistamines", "decongestants", "corticosteroids"],
    "infection": ["antibiotics", "antivirals", "antifungals"],
    "chronic": ["beta blockers", "ACE inhibitors", "statins", "insulin"],
}

# Emotional states and their expressions
EMOTIONAL_STATES = {
    "concerned": ["worried about", "concerned about", "anxious about"],
    "relieved": ["feeling better", "improved", "relieved"],
    "frustrated": ["frustrated with", "annoyed by", "tired of"],
    "hopeful": ["hoping for", "looking forward to", "optimistic about"],
}


def generate_dialogue(condition: str, symptoms: List[str]) -> Dict:
    """Generate a medical dialogue entry."""
    # Randomly select symptoms to mention
    num_symptoms = random.randint(1, len(symptoms))
    selected_symptoms = random.sample(symptoms, num_symptoms)

    # Generate patient's description
    emotional_state = random.choice(list(EMOTIONAL_STATES.keys()))
    emotional_phrase = random.choice(EMOTIONAL_STATES[emotional_state])

    # Build the dialogue text
    text = f"I'm {emotional_phrase} my {condition}. "
    text += "I've been experiencing " + ", ".join(selected_symptoms[:-1])
    if len(selected_symptoms) > 1:
        text += f" and {selected_symptoms[-1]}"
    else:
        text += selected_symptoms[0]
    text += "."

    # Add medication/treatment context
    if random.random() < 0.7:  # 70% chance to mention medication
        med_type = random.choice(list(MEDICATIONS.keys()))
        medication = random.choice(MEDICATIONS[med_type])
        text += f" I've been taking {medication} but it's not helping much."

    return {
        "text": text,
        "condition": condition,
        "symptoms": selected_symptoms,
        "emotional_state": emotional_state,
    }


def generate_dataset(num_samples: int = 10000) -> List[Dict]:
    """Generate a large medical dialogue dataset."""
    dataset = []
    conditions = list(MEDICAL_CONDITIONS.keys())

    for _ in range(num_samples):
        condition = random.choice(conditions)
        symptoms = MEDICAL_CONDITIONS[condition]
        dialogue = generate_dialogue(condition, symptoms)
        dataset.append(dialogue)

    return dataset


def save_dataset(dataset: List[Dict], file_path: str) -> None:
    """Save the generated dataset to a file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(dataset, f, indent=2)


if __name__ == "__main__":
    # Generate a large dataset
    print("Generating medical dialogue dataset...")
    dataset = generate_dataset(num_samples=10000)

    # Save the dataset
    save_dataset(dataset, "data/medical_dialogues.json")
    print(f"Dataset generated and saved to data/medical_dialogues.json")
    print(f"Total samples: {len(dataset)}")
