import requests
import os

class LLMClient:
    def __init__(self, model: str = None, endpoint: str = None):
        self.model = model or os.getenv("LLM_MODEL", "phi3")
        self.endpoint = endpoint or os.getenv("LLM_ENDPOINT", "http://localhost:11434/api/generate")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {"num_predict": max_tokens}
        }
        response = requests.post(self.endpoint, json=payload)
        response.raise_for_status()
        return response.json()["response"]