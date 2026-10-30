"""Independent Dense/BM25 recall with safe, explicit branch degradation."""

import logging
from time import perf_counter

import requests
from asgiref.sync import sync_to_async
from django.db import DatabaseError

from khoj.database.adapters import AgentAdapters, EntryAdapters
from khoj.search_type import text_search
from khoj.search_type.rank_fusion import reciprocal_rank_fusion
from khoj.search_type.retrieval_types import HybridOptions
from khoj.utils.state import SearchType

logger = logging.getLogger(__name__)


class RetrievalUnavailable(RuntimeError):
    """Retrieval cannot satisfy the response contract with available backends."""


async def query(
    raw_query,
    user,
    type=SearchType.All,
    question_embedding=None,
    max_distance=None,
    agent=None,
    options=None,
    authorization_user=None,
    original_query=None,
):
    """Materialize both branches before fusion; never relax filters on failure.

    ORM work is thread-sensitive and sequential, so no lazy queryset or database
    connection crosses parallel worker threads. Empty success remains success.
    Permission/programming errors propagate rather than masquerading as outages.
    """
    principal = authorization_user if authorization_user is not None else user
    if agent is not None and not await AgentAdapters.ais_agent_accessible(agent, principal):
        raise PermissionError("Agent is not accessible by this user")
    options = options or HybridOptions()
    dense_hits, bm25_hits = [], []
    dense_error = bm25_error = None
    start = perf_counter()
    try:
        rows = await text_search.query(
            raw_query,
            user,
            type,
            question_embedding=question_embedding,
            max_distance=max_distance,
            agent=agent,
            top_k=options.dense_top_k,
        )
        dense_hits = text_search.to_retrieval_hits(rows)
    except (DatabaseError, requests.RequestException, RuntimeError) as error:
        dense_error = error
        logger.warning("Hybrid Dense branch unavailable (%s)", error.__class__.__name__)
    dense_end = perf_counter()
    try:
        bm25_hits = await sync_to_async(EntryAdapters.search_with_bm25)(
            raw_query=raw_query,
            user=user,
            max_results=options.bm25_top_k,
            file_type_filter=text_search.search_type_to_embeddings_type[type.value],
            agent=agent,
            authorization_user=principal,
            original_query = original_query
        )
    except DatabaseError as error:
        bm25_error = error
        logger.warning("Hybrid BM25 branch unavailable (%s)", error.__class__.__name__)
    bm25_end = perf_counter()
    if dense_error is not None and bm25_error is not None:
        raise RetrievalUnavailable("Both Dense and BM25 retrieval failed") from bm25_error
    hits = reciprocal_rank_fusion(
        dense_hits,
        bm25_hits,
        k=options.rrf_k,
        dense_weight=options.dense_weight,
        bm25_weight=options.bm25_weight,
        top_k=options.fusion_top_k,
    )
    overlap = len({h.entry_id for h in dense_hits} & {h.entry_id for h in bm25_hits})
    actual_mode = "bm25" if dense_error is not None else "vector" if bm25_error is not None else "hybrid"
    logger.info(
        "Retrieval mode=%s dense_ms=%.2f bm25_ms=%.2f rrf_ms=%.2f dense_count=%d bm25_count=%d overlap=%d unique=%d fused=%d",
        actual_mode,
        (dense_end - start) * 1000,
        (bm25_end - dense_end) * 1000,
        (perf_counter() - bm25_end) * 1000,
        len(dense_hits),
        len(bm25_hits),
        overlap,
        len({h.entry_id for h in dense_hits + bm25_hits}),
        len(hits),
    )
    return hits
