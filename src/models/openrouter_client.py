"""
OpenRouter LLM Client with NVIDIA API Fallback

Usage:
    from models.openrouter_client import OpenRouterClient

    client = OpenRouterClient(api_key="your-key", model="meta-llama/llama-3.3-70b-instruct:free")
    response = client.query("What is AI?")
    print(response)
"""

import os
import time
from openai import OpenAI
from openai import RateLimitError


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class OpenRouterClient:
    """OpenRouter client wrapper with NVIDIA API fallback."""
    
    def __init__(
        self,
        api_key=None,
        nvapi_key=None,
        model="meta-llama/llama-3.3-70b-instruct:free",
        temperature=0.7,
        max_tokens=2048,
    ):
        """
        Initialize the OpenRouter client with optional NVIDIA fallback.
        
        Args:
            api_key: OpenRouter API key (uses OPENROUTER_API_KEY env var if None)
            nvapi_key: NVIDIA API key (uses NVAPI_KEY env var if None)
            model: Model name (default: "meta-llama/llama-3.3-70b-instruct:free")
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Max response length (default: 2048)
        """
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
        if nvapi_key is None:
            nvapi_key = os.getenv("NVAPI_KEY")
        
        if not api_key and not nvapi_key:
            raise ValueError(
                "No API keys provided. Set OPENROUTER_API_KEY or NVAPI_KEY env vars."
            )
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenRouter client
        self.openrouter_client = None
        if api_key:
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        
        # Initialize NVIDIA client
        self.nvapi_client = None
        self.nvapi_model = "meta/llama-3.3-70b-instruct"
        if nvapi_key:
            self.nvapi_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=nvapi_key,
            )
    
    def query(self, message, system_prompt=None, temperature=None, max_tokens=None):
        """
        Query the model with fallback support.
        
        Tries OpenRouter first, falls back to NVIDIA if rate limited.
        If both fail, sleeps for 5 minutes and retries OpenRouter.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            
        Returns:
            Response text from the model
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Try OpenRouter first
        if self.openrouter_client:
            try:
                response = self.openrouter_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                )
                return response.choices[0].message.content
            except RateLimitError:
                print("OpenRouter rate limited. Trying NVIDIA API...")
        
        # Try NVIDIA fallback
        if self.nvapi_client:
            try:
                response = self.nvapi_client.chat.completions.create(
                    model=self.nvapi_model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                )
                return response.choices[0].message.content
            except RateLimitError:
                print("NVIDIA API rate limited. Both APIs exhausted.")
        
        # Both are rate limited, sleep for 5 minutes
        print("Both APIs rate limited. Sleeping for 5 minutes (300 seconds)...")
        time.sleep(300)
        
        # Retry OpenRouter after sleep
        if self.openrouter_client:
            response = self.openrouter_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )
            return response.choices[0].message.content
        
        # If OpenRouter unavailable, retry NVIDIA
        if self.nvapi_client:
            response = self.nvapi_client.chat.completions.create(
                model=self.nvapi_model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )
            return response.choices[0].message.content
        
        raise RuntimeError("No API clients available")

