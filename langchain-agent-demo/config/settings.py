"""
Configuration management for the LangChain Agent Demo.
Loads environment variables and provides settings throughout the application.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    cohere_api_key: Optional[str] = Field(default=None, env="COHERE_API_KEY")
    tavily_api_key: Optional[str] = Field(default=None, env="TAVILY_API_KEY")
    
    # LangSmith Configuration
    langchain_tracing_v2: bool = Field(default=False, env="LANGCHAIN_TRACING_V2")
    langchain_api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="langchain-agent-demo", env="LANGCHAIN_PROJECT")
    
    # Model Settings
    default_model: str = Field(default="gpt-4o-mini", env="DEFAULT_MODEL")
    embedding_model: str = Field(default="text-embedding-ada-002", env="EMBEDDING_MODEL")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    max_tokens: int = Field(default=2000, env="MAX_TOKENS")
    
    # Project Paths
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent
    
    @property
    def data_dir(self) -> Path:
        path = self.project_root / "data"
        path.mkdir(exist_ok=True)
        return path
    
    @property
    def examples_dir(self) -> Path:
        return self.project_root / "examples"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def validate_api_keys() -> bool:
    """
    Validate that required API keys are set.
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not settings.openai_api_key:
        print("⚠️  Warning: OPENAI_API_KEY not set in .env file")
        print("   Most examples require an OpenAI API key to run.")
        print("   Copy .env.example to .env and add your API key.")
        return False
    return True


def setup_langsmith():
    """Configure LangSmith tracing if enabled."""
    if settings.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if settings.langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        if settings.langchain_project:
            os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        print("✅ LangSmith tracing enabled")


# Initialize LangSmith on import
setup_langsmith()

