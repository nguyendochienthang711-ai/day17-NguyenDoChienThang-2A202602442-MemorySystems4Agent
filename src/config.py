from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a populated LabConfig instance."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load .env if present
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()

    # Create state directory
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    data_dir = root / "data"

    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "700"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    main_provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    main_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    main_api_key = (
        os.getenv("OPENAI_API_KEY")
        if main_provider in ("openai", "custom")
        else os.getenv("GEMINI_API_KEY")
        if main_provider == "gemini"
        else os.getenv("ANTHROPIC_API_KEY")
        if main_provider == "anthropic"
        else os.getenv("OPENROUTER_API_KEY")
        if main_provider == "openrouter"
        else None
    )
    main_base_url = (
        os.getenv("CUSTOM_BASE_URL")
        if main_provider == "custom"
        else os.getenv("OLLAMA_BASE_URL")
        if main_provider == "ollama"
        else None
    )

    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", main_provider))
    judge_model_name = os.getenv("JUDGE_MODEL", main_model)

    model_config = ProviderConfig(
        provider=main_provider,
        model_name=main_model,
        temperature=0.0,
        api_key=main_api_key,
        base_url=main_base_url,
    )

    judge_config = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=main_api_key,
        base_url=main_base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model_config,
        judge_model=judge_config,
    )
