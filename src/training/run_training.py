"""
Script to run training for medical NER model.
"""

import json
from src.training.train_ner import train_medical_ner


def load_medical_dialogues(file_path: str):
    """Load and prepare medical dialogues data."""
    with open(file_path, "r") as f:
        data = json.load(f)

    # Convert to training format
    train_data = []
    for item in data:
        text = item["text"]
        words = text.split()
        labels = ["O"] * len(words)  # Initialize all as 'O'

        # Label symptoms
        for symptom in item.get("symptoms", []):
            symptom_words = symptom.lower().split()
            for i in range(len(words)):
                if words[i].lower() in symptom_words:
                    if i == 0 or labels[i - 1] == "O":
                        labels[i] = "B-SYMPTOM"
                    else:
                        labels[i] = "I-SYMPTOM"

        # Label conditions
        for condition in item.get("condition", []):
            condition_words = condition.lower().split()
            for i in range(len(words)):
                if words[i].lower() in condition_words:
                    if i == 0 or labels[i - 1] == "O":
                        labels[i] = "B-CONDITION"
                    else:
                        labels[i] = "I-CONDITION"

        train_data.append({"text": text, "labels": labels})

    return train_data


def main():
    # Load training data
    print("Loading medical dialogues data...")
    data_path = "data/medical_dialogues.json"
    train_data = load_medical_dialogues(data_path)

    # Split into train and validation
    train_size = int(0.8 * len(train_data))
    train_set = train_data[:train_size]
    val_set = train_data[train_size:]

    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")

    # Train the model
    print("Training medical NER model...")
    model, tokenizer = train_medical_ner(
        train_data=train_set,
        val_data=val_set,
        model_name="distilbert-base-uncased",
        output_dir="models/medical_ner",
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        max_length=128,
    )

    print("Training completed. Model saved to models/medical_ner")


if __name__ == "__main__":
    main()
