"""
Language model for generating empathetic responses based on detected emotions.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional, List
import os

class ResponseGenerator:
    """Generates empathetic responses based on detected emotions."""
    
    def __init__(self, model_name: str = "distilgpt2", max_length: int = 50, 
                 temperature: float = 0.7):
        """
        Initialize the response generator.
        
        Args:
            model_name: Name or path of the pretrained language model
            max_length: Maximum length of generated responses
            temperature: Temperature for sampling (higher = more creative)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Emotion-specific prompt templates
        self.emotion_prompts = {
            "angry": [
                "I notice you seem frustrated. ",
                "You look upset. ",
                "I can see you're feeling angry. "
            ],
            "disgust": [
                "I see something's bothering you. ",
                "You seem disturbed by something. ",
                "I notice you're having a negative reaction. "
            ],
            "fear": [
                "You appear concerned. ",
                "I notice you seem worried. ",
                "You look anxious. "
            ],
            "happy": [
                "I see you're in a good mood! ",
                "You look happy! ",
                "I notice you're smiling! "
            ],
            "sad": [
                "I notice you seem down. ",
                "You look a bit sad. ",
                "I can see something's troubling you. "
            ],
            "surprise": [
                "You look surprised! ",
                "I see something unexpected happened! ",
                "You seem startled! "
            ],
            "neutral": [
                "How are you feeling? ",
                "I'm here if you need anything. ",
                "Is there something I can help with? "
            ]
        }
        
    def initialize(self) -> bool:
        """
        Load the language model and tokenizer.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            print(f"Loading {self.model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            print(f"Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"Error loading language model: {e}")
            return False
    
    def _get_emotion_prompt(self, emotion: str) -> str:
        """
        Get a random prompt template for a specific emotion.
        
        Args:
            emotion: Detected emotion
            
        Returns:
            Prompt template string for the emotion
        """
        if emotion.lower() in self.emotion_prompts:
            templates = self.emotion_prompts[emotion.lower()]
            # Randomly select a template (you could also cycle through them)
            import random
            return random.choice(templates)
        else:
            return "I notice something. "
    
    def generate_response(self, emotion: str, context: Optional[str] = None) -> str:
        """
        Generate an empathetic response based on the detected emotion.
        
        Args:
            emotion: Detected emotion
            context: Optional additional context (e.g., previous conversation)
            
        Returns:
            Generated response text
        """
        if self.model is None or self.tokenizer is None:
            return "I'm not fully initialized yet."
        
        # Create prompt with emotion-specific template
        prompt = self._get_emotion_prompt(emotion)
        
        # Add context if provided
        if context:
            prompt += f"Based on our conversation, {context} "
        
        # Generate response
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Set up generation parameters
        gen_kwargs = {
            "max_length": len(input_ids[0]) + self.max_length,
            "temperature": self.temperature,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "no_repeat_ngram_size": 2,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # Generate response
        with torch.no_grad():
            output_sequences = self.model.generate(input_ids, **gen_kwargs)
        
        # Decode response
        generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        
        # Remove the prompt from the response
        response = generated_text[len(prompt):].strip()
        
        return response
    
    def add_custom_prompts(self, emotion: str, prompts: List[str]):
        """
        Add custom prompts for an emotion.
        
        Args:
            emotion: Emotion name
            prompts: List of prompt templates for the emotion
        """
        if emotion not in self.emotion_prompts:
            self.emotion_prompts[emotion] = []
            
        self.emotion_prompts[emotion].extend(prompts)


def generate_response(emotion: str, generator: ResponseGenerator, 
                     context: Optional[str] = None) -> str:
    """
    Generate an empathetic response based on the detected emotion.
    
    Args:
        emotion: Detected emotion
        generator: Initialized ResponseGenerator
        context: Optional additional context
        
    Returns:
        Generated response text
    """
    return generator.generate_response(emotion, context) 