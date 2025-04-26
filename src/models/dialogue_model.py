"""
Dialogue generation model for medical responses.
"""

from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class DialogueGenerator:
    """Generates medical dialogue responses."""

    @staticmethod
    def load(model_path: str):
        """Load pre-trained model and tokenizer."""
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return DialogueGenerator(model, tokenizer)

    def __init__(self, model, tokenizer):
        """Initialize generator."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ) -> str:
        """Generate response from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
