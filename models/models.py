from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class Source(str, Enum):
    SEC = "SEC"
    # CSA = "CSA"
    Earnings_Call_Transcripts = "EarningsCallTranscripts"

class FormType(str, Enum):
    earnings_transcript = "earnings_transcript"
    other_transcript = "other_transcript"
    _8_K = "8-K"
    _10_K = "10-K"
    _10_Q = "10-Q"
    DEFA14A = "DEFA14A"
    DEF_14A = "DEF 14A"
    S_1 = "S-1"
    S_3 = "S-3"
    _20_F = "20-F"
    _6_K = "6-K"


class DocumentMetadata(BaseModel):
    source: Optional[Source] = None
    filename: Optional[str] = None
    published_date: Optional[int] = None
    url: Optional[str] = None
    form_type: Optional[FormType] = None
    document_id: Optional[str] = None
    symbol: Optional[str] = None
    cik: Optional[str] = None
    fiscal_quarter: Optional[int] = None
    fiscal_year: Optional[int] = None
    is_xbrl: Optional[bool] = None


class DocumentChunkMetadata(DocumentMetadata):
    document_id: Optional[str] = None


class DocumentChunk(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: DocumentChunkMetadata
    embedding: Optional[List[float]] = None

    def to_dict(self):
        return self.dict()


class DocumentChunkWithScore(DocumentChunk):
    score: float

    def to_dict(self):
        return self.dict()


class Document(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[DocumentMetadata] = None


class DocumentWithChunks(Document):
    chunks: List[DocumentChunk]


class DocumentMetadataFilter(BaseModel):
    filenames: Optional[List[str]] = None
    document_ids: Optional[List[str]] = None
    form_types: Optional[List[FormType]] = None
    symbol: Optional[str] = None
    document_id: Optional[str] = None
    fiscal_quarter: Optional[int] = None
    fiscal_year: Optional[int] = None
    xbrl_only: Optional[bool] = None
    sort_order: Optional[str] = "desc"
    limit: Optional[int] = None
    user_id: Optional[int] = None
    controller: Optional[str] = None
    model: Optional[str] = None


class Query(BaseModel):
    query: str
    filter: Optional[DocumentMetadataFilter] = None
    sort_order: Optional[str] = "desc"
    limit: Optional[int] = None
    top_k: Optional[int] = 15


class QueryWithEmbedding(Query):
    embedding: List[float]


class QueryResult(BaseModel):
    query: str
    results: List[DocumentChunkWithScore]

    def to_dict(self):
        return {
            "query": self.query,
            "results": [result.to_dict() for result in self.results]
        }
