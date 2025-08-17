import requests
import os

class LLMClient:
    def __init__(self, model: str = None, endpoint: str = None):
        self.model = model or os.getenv("LLM_MODEL", "phi3")
        # Updated to work in Docker compose environment
        self.endpoint = endpoint or os.getenv("LLM_ENDPOINT", "http://ollama:11434/api/generate")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "options": {"num_predict": max_tokens},
                "stream": False
            }
            response = requests.post(self.endpoint, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            # Fallback error message if local LLM is unavailable
            return f"Local LLM unavailable: {str(e)}. Using fallback response."