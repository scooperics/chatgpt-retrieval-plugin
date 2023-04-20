from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class Source(str, Enum):
    _8_K = "8-K"
    _10_K = "10-K"
    _10_K_table = "10-K-table"
    _1Q = "1Q"
    _1Q_table = "1Q-table"
    _2Q = "2Q"
    _2Q_table = "2Q-table"
    _3Q = "3Q"
    _3Q_table = "3Q-table"
    DEFA14A = "DEFA14A"
    DEF_14A = "DEF 14A"
    S_1 = "S-1"
    S_3 = "S-3"
    earnings_transcript = "earnings_transcript"

class DocumentMetadata(BaseModel):
    source: Optional[Source] = None
    source_id: Optional[str] = None
    url: Optional[str] = None
    created_at: Optional[str] = None
    author: Optional[str] = None
    symbol: Optional[str] = None
    cik: Optional[str] = None
    description: Optional[str] = None


class DocumentChunkMetadata(DocumentMetadata):
    document_id: Optional[str] = None


class DocumentChunk(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: DocumentChunkMetadata
    embedding: Optional[List[float]] = None


class DocumentChunkWithScore(DocumentChunk):
    score: float


class Document(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[DocumentMetadata] = None


class DocumentWithChunks(Document):
    chunks: List[DocumentChunk]


class DocumentMetadataFilter(BaseModel):
    document_id: Optional[str] = None
    source: Optional[Source] = None
    source_id: Optional[str] = None
    author: Optional[str] = None
    start_date: Optional[str] = None  # any date string format
    end_date: Optional[str] = None  # any date string format
    symbol: Optional[str] = None
    cik: Optional[str] = None


class Query(BaseModel):
    query: str
    filter: Optional[DocumentMetadataFilter] = None
    top_k: Optional[int] = 3


class QueryWithEmbedding(Query):
    embedding: List[float]


class QueryResult(BaseModel):
    query: str
    results: List[DocumentChunkWithScore]
