"""Pydantic request/response schemas for the API surface."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    source: str = Field(..., description="Stable identifier for the doc (path, URL, or title).")
    text: str = Field(..., description="Raw document text to be chunked and embedded.")


class IngestRequest(BaseModel):
    documents: list[Document]


class IngestResponse(BaseModel):
    documents: int
    chunks_upserted: int
    collection: str


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1, le=50)
    stream: bool = False


class Citation(BaseModel):
    id: str
    source: str
    chunk_index: int
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
