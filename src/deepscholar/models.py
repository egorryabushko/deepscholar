from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


class Paper(BaseModel):
    id: str
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    source: Literal["arxiv", "openalex", "demo"]
    url: HttpUrl
    pdf_url: HttpUrl | None = None
    doi: str | None = None


class Chunk(BaseModel):
    paper_id: str
    paper_title: str
    text: str
    index: int
    source_url: HttpUrl
    score: float = 0.0
    page: int | None = None
    kind: Literal["pdf", "abstract", "demo_note"] = "abstract"


class Citation(BaseModel):
    id: str
    paper_id: str
    title: str
    url: HttpUrl
    quote: str
    page: int | None = None
    kind: Literal["pdf", "abstract", "demo_note"]


class TextPage(BaseModel):
    text: str
    page: int | None = None
    kind: Literal["pdf", "abstract", "demo_note"] = "abstract"


class WarningMessage(BaseModel):
    stage: str
    source: str
    message: str


class ResearchRequest(BaseModel):
    question: str = Field(min_length=5, max_length=800)
    max_papers: int = Field(default=4, ge=1, le=10)
    sources: list[Literal["arxiv", "openalex"]] = Field(
        default_factory=lambda: ["arxiv", "openalex"], min_length=1, max_length=2
    )

    @field_validator("question")
    @classmethod
    def meaningful_question(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 5:
            raise ValueError("Please enter a research question of at least 5 characters")
        return value

    @field_validator("sources")
    @classmethod
    def unique_sources(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class TraceStep(BaseModel):
    name: str
    status: Literal["ok", "partial", "failed"]
    duration_ms: int
    detail: str


class ResearchResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    papers: list[Paper]
    trace: list[TraceStep]
    metrics: dict[str, float]
    mode: Literal["demo", "live"]
    provider: str
    warnings: list[WarningMessage] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
