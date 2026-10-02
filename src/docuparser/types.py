from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Segment:
    """Raw extracted unit from a PDF (a page's text, or a description of one image)."""
    text: str
    page: int
    kind: str  # "text" | "image" | "ocr_page"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity, higher = better


@dataclass
class Answer:
    question: str
    answer: str | None
    sources: list[RetrievedChunk]
