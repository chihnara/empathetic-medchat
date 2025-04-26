"""
Script to test the trained medical NER model.
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from src.training.train_ner import MedicalNERDataset
import re


def load_model(model_path):
    """Load the trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    return model, tokenizer


def clean_entity(entity):
    """Clean up entity text by removing punctuation and extra spaces."""
    # Remove punctuation at the end of words
    entity = re.sub(r"[.,!?](\s|$)", r"\1", entity)
    # Remove extra spaces
    entity = " ".join(entity.split())
    return entity


def is_valid_entity(entity):
    """Check if the entity is valid (not just stop words or common words)."""
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "his",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "the",
        "their",
        "include",
        "reports",
        "experiencing",
        "having",
        "feeling",
        "been",
        "have",
    }

    words = entity.lower().split()
    return not all(word in stop_words for word in words)


def predict_entities(text, model, tokenizer):
    """Predict medical entities in the given text."""
    # Tokenize the text
    tokens = text.split()
    tokenized = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    # Get predictions
    with torch.no_grad():
        outputs = model(**tokenized)
        predictions = torch.argmax(outputs.logits, dim=2)
        confidence = torch.softmax(outputs.logits, dim=2)

    # Convert predictions to labels
    word_ids = tokenized.word_ids()
    previous_word_idx = None
    label_sequence = []
    confidence_sequence = []

    for idx, (pred_idx, conf) in enumerate(zip(predictions[0], confidence[0])):
        word_idx = word_ids[idx]

        # Skip special tokens
        if word_idx is None:
            continue

        # Only append label for the first token of each word
        if word_idx != previous_word_idx:
            label = MedicalNERDataset.id2label[pred_idx.item()]
            confidence_score = conf[pred_idx].item()
            label_sequence.append((label, confidence_score))
            previous_word_idx = word_idx

    # Combine tokens and labels
    results = []
    current_entity = []
    current_label = ""
    current_confidence = 0.0

    for token, (label, confidence) in zip(tokens, label_sequence):
        if label.startswith("B-"):
            if current_entity:
                entity_text = clean_entity(" ".join(current_entity))
                if is_valid_entity(entity_text):
                    results.append((entity_text, current_label, current_confidence))
            current_entity = [token]
            current_label = label[2:]
            current_confidence = confidence
        elif label.startswith("I-") and current_entity and label[2:] == current_label:
            current_entity.append(token)
            current_confidence = (current_confidence + confidence) / 2
        else:
            if current_entity:
                entity_text = clean_entity(" ".join(current_entity))
                if is_valid_entity(entity_text):
                    results.append((entity_text, current_label, current_confidence))
            current_entity = []
            current_label = ""
            current_confidence = 0.0

    if current_entity:
        entity_text = clean_entity(" ".join(current_entity))
        if is_valid_entity(entity_text):
            results.append((entity_text, current_label, current_confidence))

    # Filter out low confidence predictions
    results = [(entity, label, conf) for entity, label, conf in results if conf > 0.5]

    return results


def main():
    # Load the trained model
    print("Loading trained model...")
    model_path = "src/models/medical_ner"
    model, tokenizer = load_model(model_path)

    # Test cases
    test_cases = [
        "I have been experiencing severe headaches and dizziness for the past week.",
        "My chronic back pain has been getting worse, and I'm also feeling nauseous.",
        "I was diagnosed with high blood pressure and diabetes last year.",
        "The patient is experiencing chest pain and shortness of breath.",
        "I have a fever of 101 degrees and a persistent cough.",
        "The patient reports joint pain in their knees and lower back.",
        "She has been taking antibiotics for her sinus infection.",
        "His symptoms include fatigue, weight loss, and night sweats.",
    ]

    # Test the model
    print("\nTesting model on sample medical texts:")
    print("-" * 80)

    for text in test_cases:
        print(f"\nText: {text}")
        entities = predict_entities(text, model, tokenizer)
        if entities:
            print("Identified entities:")
            for entity, label, confidence in entities:
                print(f"  - {entity} ({label}, confidence: {confidence:.2f})")
        else:
            print("No medical entities identified.")
        print("-" * 80)


if __name__ == "__main__":
    main()
