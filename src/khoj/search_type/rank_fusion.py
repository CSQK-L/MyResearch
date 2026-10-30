"""Deterministic rank-only fusion, independent of databases and inference."""

import math
from numbers import Real
from typing import List, Sequence

from khoj.search_type.retrieval_types import RetrievalHit


def reciprocal_rank_fusion(
    dense_hits: Sequence[RetrievalHit],
    bm25_hits: Sequence[RetrievalHit],
    k: float = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    top_k: int = 40,
) -> List[RetrievalHit]:
    """Use list positions as 1-based ranks; first occurrence wins per branch.

    Duplicates retain their original positions but contribute only once. New
    hits preserve branch evidence without mutating either input.
    """
    for name, value in (("k", k), ("dense_weight", dense_weight), ("bm25_weight", bm25_weight)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and numeric")
        if value < 0 or (name == "k" and value == 0):
            raise ValueError(f"{name} must be {'positive' if name == 'k' else 'non-negative'}")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")

    merged = {}
    for branch, weight, source in (("dense", dense_weight, dense_hits), ("bm25", bm25_weight, bm25_hits)):
        seen = set()
        for rank, hit in enumerate(source, 1):
            if hit.entry_id in seen:
                continue
            seen.add(hit.entry_id)
            fused = merged.setdefault(hit.entry_id, RetrievalHit(entry_id=hit.entry_id))
            if branch == "dense":
                fused.dense_rank = rank
                fused.cosine_distance = hit.cosine_distance
            else:
                fused.bm25_rank = rank
                fused.bm25_score = hit.bm25_score
            fused.rrf_score += weight / (k + rank)
    return sorted(merged.values(), key=lambda hit: (-hit.rrf_score, hit.entry_id))[:top_k]
