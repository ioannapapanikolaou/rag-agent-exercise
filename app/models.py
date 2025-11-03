from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., description="User question or query")
    k: int = Field(5, description="Number of chunks to retrieve")


class Citation(BaseModel):
    source: str = Field(..., description="Document source path or identifier")
    start: int = Field(..., description="Start character offset in source")
    end: int = Field(..., description="End character offset in source (exclusive)")


class Metrics(BaseModel):
    latency_ms: float
    retrieved_k: int
    route: str
    extra: Optional[Dict[str, Any]] = None


class AnswerResponse(BaseModel):
    answer: str
    citations: List[Citation]
    sources: List[str]
    metrics: Optional[Metrics] = None


class IngestStats(BaseModel):
    documents: int
    chunks: int
    bytes_read: int
    sources: List[str]
