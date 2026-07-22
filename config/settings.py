"""
Central application settings.

All configuration flows through this module. Values are read from the .env
file at the project root, falling back to the defaults below. No other module
should access os.environ directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Typed, validated application configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    hf_token: str = Field(..., description="Hugging Face API token")

    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    llm_base_url: str = "https://router.huggingface.co/v1"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=512, gt=0)

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cuda"
    embedding_batch_size: int = Field(default=64, gt=0)

    retrieval_top_k: int = Field(default=5, gt=0)
    retrieval_score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    dataset_name: str = (
        "flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl"
    )
    # NoDecode disables the automatic JSON parsing that pydantic-settings
    # applies to complex types, allowing the comma-separated form below.
    dataset_configs: Annotated[list[str], NoDecode] = ["datascience", "ai", "stats"]
    max_documents: int = Field(default=5000, gt=0)

    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    vectorstore_dir: Path = PROJECT_ROOT / "data" / "vectorstore"
    reports_dir: Path = PROJECT_ROOT / "reports"

    @field_validator("dataset_configs", mode="before")
    @classmethod
    def split_configs(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("embedding_device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        allowed = {"cuda", "cpu", "mps"}
        if value not in allowed:
            raise ValueError(f"embedding_device must be one of {allowed}")
        return value


settings = Settings()