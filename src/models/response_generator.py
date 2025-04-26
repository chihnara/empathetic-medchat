"""
Response generator for medical dialogue with empathy.
"""

from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class ResponseGenerator:
    """Generates empathetic medical responses."""

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the response generator."""
        self.device = device
        self.model_name = model_name

        # Initialize text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device=-1,  # Force CPU
            max_length=100,
            num_return_sequences=1,
        )

    def generate_response(
        self, context: Dict, empathy_level: str, max_length: int = 100
    ) -> str:
        """Generate a response based on context and empathy level."""
        try:
            # Construct prompt from context
            prompt = self._construct_prompt(context, empathy_level)

            # Generate response
            generated = self.generator(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

            # Extract generated text
            response = generated[0]["generated_text"]

            # Clean up response
            response = self._clean_response(response, prompt)

            return response

        except Exception as e:
            print(f"Error in response generation: {str(e)}")
            return "I apologize, but I'm having trouble generating a response at the moment."

    def _construct_prompt(self, context: Dict, empathy_level: str) -> str:
        """Construct prompt from context and empathy level."""
        # Extract medical context
        medical_context = context.get("medical", {})
        symptoms = medical_context.get("symptoms", [])
        conditions = medical_context.get("conditions", [])

        # Extract emotional context
        emotional_context = context.get("emotional", {})
        emotions = emotional_context.get("emotions", [])

        # Construct prompt
        prompt = f"Patient is experiencing {', '.join(symptoms)}"
        if conditions:
            prompt += f" with {', '.join(conditions)}"
        if emotions:
            prompt += f" and feeling {', '.join(emotions)}"
        prompt += f". Generate a {empathy_level} empathy response:"

        return prompt

    def _clean_response(self, response: str, prompt: str) -> str:
        """Clean up generated response."""
        # Remove prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()

        # Remove any remaining special tokens
        response = response.replace("<|endoftext|>", "").strip()

        return response
