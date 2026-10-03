from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents."""

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map provider names and aliases to normalized strings."""
    alias = value.strip().lower()
    mapping = {
        "openai": "openai",
        "custom": "custom",
        "gemini": "gemini",
        "google": "gemini",
        "google-genai": "gemini",
        "google_genai": "gemini",
        "anthropic": "anthropic",
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "ollama": "ollama",
        "openrouter": "openrouter",
        "open_router": "openrouter",
    }
    if alias in mapping:
        return mapping[alias]
    return alias


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate the real chat model for the selected provider."""
    provider = normalize_provider(config.provider)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )
        except ImportError as exc:
            raise ImportError("Please install langchain-openai to use the OpenAI provider.") from exc

    elif provider == "custom":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
                base_url=config.base_url,
            )
        except ImportError as exc:
            raise ImportError("Please install langchain-openai to use the custom OpenAI-compatible provider.") from exc

    elif provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=config.model_name,
                temperature=config.temperature,
                google_api_key=config.api_key,
            )
        except ImportError as exc:
            raise ImportError("Please install langchain-google-genai to use the Gemini provider.") from exc

    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )
        except ImportError as exc:
            raise ImportError("Please install langchain-anthropic to use the Anthropic provider.") from exc

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=config.model_name,
                temperature=config.temperature,
                base_url=config.base_url,
            )
        except ImportError as exc:
            raise ImportError("Please install langchain-ollama to use the Ollama provider.") from exc

    elif provider == "openrouter":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
                base_url=config.base_url or "https://openrouter.ai/api/v1",
            )
        except ImportError as exc:
            raise ImportError("Please install langchain-openai to use the OpenRouter provider.") from exc

    raise ValueError(f"Unsupported provider: {config.provider}")
