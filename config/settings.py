"""
Central application settings.

All configuration flows through this single module. Values are read from
the .env file at the project root, with the defaults below used for any
variable that is not set. Nothing else in the codebase should call
os.environ directly.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# settings.py lives in config/, so the project root is one level up.
# parents[0] would be config/, parents[1] is the root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Typed, validated application configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # HF_TOKEN in .env -> hf_token here
        extra="ignore",         # unknown .env vars are tolerated
    )

    # ---- Secrets ----
    # Ellipsis (...) marks the field as REQUIRED. If HF_TOKEN is missing
    # from .env, construction fails immediately with a clear message --
    # far better than a confusing 401 twenty minutes into a run.
    hf_token: str = Field(..., description="Hugging Face API token")

    # ---- LLM ----
    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    llm_base_url: str = "https://router.huggingface.co/v1"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=512, gt=0)

    # ---- Embeddings ----
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cuda"
    embedding_batch_size: int = Field(default=64, gt=0)

    # ---- Retrieval ----
    retrieval_top_k: int = Field(default=5, gt=0)
    retrieval_score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # ---- Dataset ----
    dataset_name: str = (
        "flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl"
    )
    dataset_configs: list[str] = ["datascience", "ai", "stats"]
    max_documents: int = Field(default=5000, gt=0)

    # ---- Paths (derived, not read from .env) ----
    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    vectorstore_dir: Path = PROJECT_ROOT / "data" / "vectorstore"
    reports_dir: Path = PROJECT_ROOT / "reports"

    # A validator transforms or checks a value as it is loaded.
    # mode="before" runs on the raw input, so we receive the plain
    # comma-separated string from .env before pydantic tries to
    # coerce it into a list.
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


# A single shared instance, imported everywhere:
#     from config.settings import settings
# Reading and validating .env happens exactly once, at import time.
settings = Settings()