"""Central configuration. All values can be set via env vars (prefix DOCUPARSER_) or a .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DOCUPARSER_", extra="ignore")

    # storage
    data_dir: Path = Path("data")
    chroma_mode: Literal["persistent", "http"] = "persistent"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    collection_prefix: str = "docuparser"

    # embeddings
    embedding_provider: Literal["huggingface", "openai", "fake"] = "huggingface"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = ""
    embedding_device: str = "cpu"

    # LLM
    llm_provider: Literal["openai", "huggingface", "none"] = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 700

    # OpenAI (or any OpenAI-compatible server)
    openai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("DOCUPARSER_OPENAI_API_KEY", "OPENAI_API_KEY")
    )
    openai_base_url: str | None = None

    # images
    image_mode: Literal["none", "ocr", "caption", "both"] = "caption"
    caption_provider: Literal["huggingface", "openai", "fake"] = "huggingface"
    caption_model: str = "Salesforce/blip-image-captioning-base"
    vision_model: str = "gpt-4o-mini"
    min_image_side: int = 100  # skip icons/logos smaller than this (px)
    ocr_scanned_pages: bool = True  # OCR pages that have no text layer

    # chunking / retrieval
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    min_score: float = 0.0  # similarity threshold (0..1); 0 disables

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()
