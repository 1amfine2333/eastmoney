from abc import ABC, abstractmethod
import os
import sys
from google import genai
from openai import OpenAI

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import (
    GEMINI_API_KEY, GEMINI_MODEL, GEMINI_API_ENDPOINT,
    LLM_PROVIDER,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
)

class BaseLLMClient(ABC):
    @abstractmethod
    def generate_content(self, prompt: str) -> str:
        pass

class GoogleGeminiClient(BaseLLMClient):
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        api_endpoint = os.getenv("GEMINI_API_ENDPOINT")
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")
        
        client_kwargs = {"api_key": api_key}
        
        # Configure custom endpoint if provided
        if api_endpoint:
            client_kwargs["http_options"] = {"base_url": api_endpoint, "api_version": "v1alpha"}
            
        self.client = genai.Client(**client_kwargs)
        self.model_name = model

    def generate_content(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            if response.text:
                return response.text
            return "Error: No text returned from model."
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            return f"Error: Could not generate analysis. Details: {str(e)}"

class OpenAIClient(BaseLLMClient):
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("OPENAI_MODEL", "gpt-4o")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")
        
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
            
        self.client = OpenAI(**client_kwargs)
        self.model_name = model

    def generate_content(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating content with OpenAI: {e}")
            return f"Error: Could not generate analysis. Details: {str(e)}"

def get_llm_client() -> BaseLLMClient:
    """
    Factory function to return the configured LLM client.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    if provider == "gemini":
        return GoogleGeminiClient()
    elif provider == "openai" or provider == "openai_compatible":
        return OpenAIClient()
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
