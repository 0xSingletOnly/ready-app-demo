# models/llm.py
from dotenv import load_dotenv
import os
from mistralai import Mistral
from langchain.llms.base import LLM
from typing import Any, Dict, List, Mapping, Optional
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class MistralLLM(LLM):
    """LangChain wrapper for Mistral AI API."""
    
    model_name: str = "ministral-3b-latest"
    temperature: float = 0.1
    max_tokens: int = 1024
    client: Any = None
    
    def __init__(
        self, 
        model_name: str = "ministral-3b-latest", 
        temperature: float = 0.1,
        max_tokens: int = 1024
    ):
        """Initialize Mistral LLM wrapper."""
        super().__init__()
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = Mistral(api_key=api_key)
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the Mistral API."""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            response = self.client.chat.complete(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Mistral API: {e}")
            raise
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "mistral-api"

def get_mistral_model():
    """Initialize and return a Mistral LLM instance."""
    return MistralLLM(
        model_name="ministral-3b-latest",
        temperature=0.1,
        max_tokens=1024
    )