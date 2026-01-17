from dataclasses import dataclass
from typing import NewType

MovieId = NewType("MovieId", int)
Token = NewType("Token", str)
TokenCount = NewType("TokenCount", int)
ChunkIndex = NewType("ChunkIndex", int)


@dataclass
class Movie:
    id: MovieId
    title: str
    description: str


@dataclass
class Bm25Result:
    movie_id: MovieId
    score: float


@dataclass
class SemanticResult:
    movie_id: MovieId
    title: str
    description: str
    score: float


@dataclass
class ChunkMetadata:
    movie_index: int
    chunk_index: int
    total_chunks: int


@dataclass
class ChunkScore:
    movie_index: int
    chunk_index: ChunkIndex
    score: float


@dataclass
class ChunkSearchResult:
    movie_id: MovieId
    title: str
    document: str
    score: float
    movie_index: int
    best_chunk_index: ChunkIndex


@dataclass
class HybridWeightedResult:
    movie_id: MovieId
    title: str
    document: str
    bm25_score: float
    semantic_score: float
    score: float


@dataclass
class HybridRRFResult:
    movie_id: MovieId
    title: str
    document: str
    bm25_rank: int
    semantic_rank: int
    rrf_score: float
    rerank_score: float
