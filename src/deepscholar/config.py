from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DEEPSCHOLAR_", env_file=".env", extra="ignore")

    offline: bool = True
    llm_provider: Literal["extractive", "huggingface", "local"] = "extractive"
    hf_model: str = "Qwen/Qwen2.5-7B-Instruct"
    hf_token: str | None = Field(default=None, repr=False)
    hf_provider: str = "auto"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str | None = Field(default=None, repr=False)
    llm_model: str | None = None
    max_output_tokens: int = Field(default=900, ge=16, le=2048)
    evidence_limit: int = Field(default=6, ge=1, le=6)
    openalex_api_key: str | None = Field(default=None, repr=False)
    request_timeout: float = Field(default=20.0, gt=0, le=120)
    llm_timeout: float = Field(default=60.0, gt=0, le=300)
    workflow_timeout: float = Field(default=150.0, gt=0, le=600)
    max_papers: int = Field(default=6, ge=1, le=10)
    max_pdf_bytes: int = Field(default=20_000_000, ge=1, le=50_000_000)
    max_pdf_pages: int = Field(default=30, ge=1, le=100)
    chunk_chars: int = Field(default=1_400, ge=200, le=4000)
    chunk_overlap: int = Field(default=180, ge=0)
    pdf_hosts: list[str] = Field(default_factory=lambda: ["arxiv.org", "export.arxiv.org"])

    @model_validator(mode="after")
    def validate_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_chars:
            raise ValueError("chunk_overlap must be smaller than chunk_chars")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
