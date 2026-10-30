"""Internal retrieval contracts shared by Dense, BM25 and rank fusion."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RetrievalMode(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


@dataclass
class RetrievalHit:
    """One Entry primary key, with independent scores and 1-based retriever ranks.

    Entry uses Django's integer BigAutoField. Unlike corpus_id, its primary key
    uniquely identifies each chunk. Missing retriever evidence stays None;
    rrf_score is fusion evidence and is not a SearchResponse score.
    """

    entry_id: int
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    cosine_distance: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: float = 0.0


@dataclass
class RetrievalOptions:
    """Legacy Dense recall options, retained for positional/default compatibility."""

    mode: RetrievalMode = RetrievalMode.VECTOR
    dense_top_k: int = 10

    def __post_init__(self) -> None:
        self.mode = RetrievalMode(self.mode)
        if isinstance(self.dense_top_k, bool) or not isinstance(self.dense_top_k, int) or self.dense_top_k <= 0:
            raise ValueError("dense_top_k must be a positive integer")


@dataclass(frozen=True)
class HybridOptions:
    """Independent recall budgets and fusion controls, shared by all callers."""

    dense_top_k: int = 40
    bm25_top_k: int = 40
    rrf_k: float = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    fusion_top_k: int = 40

    def __post_init__(self):
        from khoj.search_type.rank_fusion import reciprocal_rank_fusion

        for value in (self.dense_top_k, self.bm25_top_k, self.fusion_top_k):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("retrieval budgets must be positive integers")
        reciprocal_rank_fusion([], [], self.rrf_k, self.dense_weight, self.bm25_weight, self.fusion_top_k)
