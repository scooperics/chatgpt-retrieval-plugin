from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class Source(str, Enum):
    SEC = "SEC"
    CSA = "CSA"
    Earnings_Call_Transcripts = "EarningsCallTranscripts"

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

class DocumentSection(str, Enum):
    _1 = '1'
    _1A = '1A'
    _1B = '1B'
    _2 = '2'
    _3 = '3'
    _4 = '4'
    _5 = '5'
    _6 = '6'
    _7 = '7'
    _7A = '7A'
    _8 = '8'
    _9 = '9'
    _9A = '9A'
    _9B = '9B'
    _10 = '10'
    _11 = '11'
    _12 = '12'
    _13 = '13'
    _14 = '14'
    _15 = '15'
    part1item1 = 'part1item1'
    part1item2 = 'part1item2'
    part1item3 = 'part1item3'
    part1item4 = 'part1item4'
    part2item1 = 'part2item1'
    part2item1a = 'part2item1a'
    part2item2 = 'part2item2'
    part2item3 = 'part2item3'
    part2item4 = 'part2item4'
    part2item5 = 'part2item5'
    part2item6 = 'part2item6'
    # _1_1 = '1-1'
    # _1_2 = '1-2'
    # _1_3 = '1-3'
    # _1_4 = '1-4'
    # _2_1 = '2-1'
    # _2_2 = '2-2'
    # _2_3 = '2-3'
    # _2_4 = '2-4'
    # _2_5 = '2-5'
    # _2_6 = '2-6'
    # _3_1 = '3-1'
    # _3_2 = '3-2'
    # _3_3 = '3-3'
    # _4_1 = '4-1'
    # _4_2 = '4-2'
    # _5_1 = '5-1'
    # _5_2 = '5-2'
    # _5_3 = '5-3'
    # _5_4 = '5-4'
    # _5_5 = '5-5'
    # _5_6 = '5-6'
    # _5_7 = '5-7'
    # _5_8 = '5-8'
    # _6_1 = '6-1'
    # _6_2 = '6-2'
    # _6_3 = '6-3'
    # _6_4 = '6-4'
    # _6_5 = '6-5'
    # _6_6 = '6-6'
    # _6_10 = '6-10'
    # _7_1 = '7-1'
    # _8_1 = '8-1'
    # _9_1 = '9-1'
    # signature = 'signature'

class DocumentMetadata(BaseModel):
    source: Optional[Source] = None
    filename: Optional[str] = None
    published_date: Optional[int] = None
    url: Optional[str] = None
    form_type: Optional[FormType] = None
    document_section: Optional[DocumentSection] = None
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
    document_sections: Optional[List[DocumentSection]] = None
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
