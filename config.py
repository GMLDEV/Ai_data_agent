import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List

class Settings(BaseSettings):
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    
    # Execution limits
    max_execution_time: int = 180  # 3 minutes
    max_retries: int = 3
    
    # File limits
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    supported_file_types: List[str] = ['.csv', '.txt', '.png', '.jpg', '.jpeg']
    
    # Sandbox limits
    sandbox_memory_limit: int = 1024 * 1024 * 1024  # 1GB
    sandbox_cpu_limit: float = 2.0
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # LLM settings
    llm_model: str = "phi3"
    llm_endpoint: str = "http://localhost:11434/api/generate"
    
    class Config:
        env_file = ".env"

settings = Settings()