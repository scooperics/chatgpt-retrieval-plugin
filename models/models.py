from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class Source(str, Enum):
    SEC = "SEC"
    CSA = "CSA"
    Earnings_Call_Transcripts = "Earnings Call Transcripts"

class FormType(str, Enum):
    _8_K = "8-K"
    _10_K = "10-K"
    _10_K_table = "10-K-table"
    _10_Q = "10-Q"
    _10_Q_table = "10-Q-table"
    DEFA14A = "DEFA14A"
    DEF_14A = "DEF 14A"
    S_1 = "S-1"
    S_3 = "S-3"
    earnings_transcript = "earnings_transcript"


class DocumentMetadata(BaseModel):
    source: Optional[Source] = None
    filename: Optional[str] = None
    published_date: Optional[int] = None
    url: Optional[str] = None
    form_type: Optional[FormType] = None
    company_name: Optional[str] = None
    document_id: Optional[str] = None
    component_number: Optional[int] = None
    symbol: Optional[str] = None
    cik: Optional[str] = None
    fiscal_quarter: Optional[int] = None
    fiscal_year: Optional[int] = None


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
    filenames: Optional[List[str]] = None
    source: Optional[Source] = None
    form_types: Optional[List[FormType]] = None
    company_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    symbol: Optional[str] = None
    cik: Optional[str] = None
    fiscal_quarter: Optional[int] = None
    fiscal_year: Optional[int] = None


class Query(BaseModel):
    query: str
    filter: Optional[DocumentMetadataFilter] = None
    top_k: Optional[int] = 10


class QueryWithEmbedding(Query):
    embedding: List[float]


class QueryResult(BaseModel):
    query: str
    results: List[DocumentChunkWithScore]
