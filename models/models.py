from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class Source(str, Enum):
    sec = "SEC"
    csa = "CSA"
    earnings_call_transcripts = "Earnings Call Transcripts"

class FormType(str, Enum):
    _8_K = "8-K"
    _10_K = "10-K"
    _10_K_table = "10-K"
    _10_Q = "10-Q"
    DEFA14A = "DEFA14A"
    DEF_14A = "DEF 14A"
    S_1 = "S-1"
    S_3 = "S-3"
    earnings_transcript = "earnings_transcript"


class DocumentMetadata(BaseModel):
    source: Optional[Source] = None
    filename: Optional[str] = None
    url: Optional[str] = None
    form_type: Optional[FormType] = None
    document_section: Optional[str] = None
    company_name: Optional[str] = None
    document_id: Optional[str] = None
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
    document_id: Optional[str] = None
    source: Optional[Source] = None
    form_type: Optional[FormType] = None
    company_name: Optional[str] = None
    document_section: Optional[str] = None
    # start_date: Optional[str] = None  # any date string format
    # end_date: Optional[str] = None  # any date string format
    symbol: Optional[str] = None
    cik: Optional[str] = None
    fiscal_quarter: Optional[int] = None
    fiscal_year: Optional[int] = None


class Query(BaseModel):
    query: str
    filter: Optional[DocumentMetadataFilter] = None
    top_k: Optional[int] = 3


class QueryWithEmbedding(Query):
    embedding: List[float]


class QueryResult(BaseModel):
    query: str
    results: List[DocumentChunkWithScore]
