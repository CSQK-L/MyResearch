import logging
import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Type, Union

import requests
import torch
from asgiref.sync import sync_to_async
from pgvector.django import CosineDistance
from sentence_transformers import util

from khoj.database.adapters import EntryAdapters, get_default_search_model
from khoj.database.models import Agent, KhojUser
from khoj.database.models import Entry as DbEntry
from khoj.processor.content.text_to_entries import TextToEntries
from khoj.search_type.retrieval_config import retrieval_query_text
from khoj.search_type.retrieval_types import RetrievalHit, RetrievalMode, RetrievalOptions
from khoj.utils import state
from khoj.utils.helpers import get_absolute_path, timer
from khoj.utils.jsonl import load_jsonl
from khoj.utils.models import BaseEncoder
from khoj.utils.rawconfig import Entry, SearchResponse
from khoj.utils.state import SearchType

logger = logging.getLogger(__name__)

search_type_to_embeddings_type = {
    SearchType.Org.value: DbEntry.EntryType.ORG,
    SearchType.Markdown.value: DbEntry.EntryType.MARKDOWN,
    SearchType.Plaintext.value: DbEntry.EntryType.PLAINTEXT,
    SearchType.Pdf.value: DbEntry.EntryType.PDF,
    SearchType.Github.value: DbEntry.EntryType.GITHUB,
    SearchType.Notion.value: DbEntry.EntryType.NOTION,
    SearchType.All.value: None,
}


def extract_entries(jsonl_file) -> List[Entry]:
    "Load entries from compressed jsonl"
    return list(map(Entry.from_dict, load_jsonl(jsonl_file)))


def compute_embeddings(
    entries_with_ids: List[Tuple[int, Entry]],
    bi_encoder: BaseEncoder,
    embeddings_file: Path,
    regenerate=False,
    normalize=True,
):
    "Compute (and Save) Embeddings or Load Pre-Computed Embeddings"
    new_embeddings = torch.tensor([], device=state.device)
    existing_embeddings = torch.tensor([], device=state.device)
    create_index_msg = ""
    # Load pre-computed embeddings from file if exists and update them if required
    if embeddings_file.exists() and not regenerate:
        corpus_embeddings: torch.Tensor = torch.load(get_absolute_path(embeddings_file), map_location=state.device)
        logger.debug(f"Loaded {len(corpus_embeddings)} text embeddings from {embeddings_file}")
    else:
        corpus_embeddings = torch.tensor([], device=state.device)
        create_index_msg = " Creating index from scratch."

    # Encode any new entries in the corpus and update corpus embeddings
    new_entries = [entry.compiled for id, entry in entries_with_ids if id == -1]
    if new_entries:
        logger.info(f"📩 Indexing {len(new_entries)} text entries.{create_index_msg}")
        new_embeddings = bi_encoder.encode(
            new_entries, convert_to_tensor=True, device=state.device, show_progress_bar=True
        )

    # Extract existing embeddings from previous corpus embeddings
    existing_entry_ids = [id for id, _ in entries_with_ids if id != -1]
    if existing_entry_ids:
        existing_embeddings = torch.index_select(
            corpus_embeddings, 0, torch.tensor(existing_entry_ids, device=state.device)
        )

    # Set corpus embeddings to merger of existing and new embeddings
    corpus_embeddings = torch.cat([existing_embeddings, new_embeddings], dim=0)
    if normalize:
        # Normalize embeddings for faster lookup via dot product when querying
        corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

    # Save regenerated or updated embeddings to file
    torch.save(corpus_embeddings, embeddings_file)
    logger.info(f"📩 Saved computed text embeddings to {embeddings_file}")

    return corpus_embeddings


def load_embeddings(
    embeddings_file: Path,
):
    "Load pre-computed embeddings from file if exists and update them if required"
    if embeddings_file.exists():
        corpus_embeddings: torch.Tensor = torch.load(get_absolute_path(embeddings_file), map_location=state.device)
        logger.debug(f"Loaded {len(corpus_embeddings)} text embeddings from {embeddings_file}")
        return util.normalize_embeddings(corpus_embeddings)

    return None


async def query(
    raw_query: str,
    user: Optional[KhojUser],
    type: SearchType = SearchType.All,
    question_embedding: Union[torch.Tensor, None] = None,
    max_distance: Optional[float] = None,
    agent: Optional[Agent] = None,
    top_k: int = 10,
) -> List[DbEntry]:
    "Search for entries that answer the query"

    RetrievalOptions(dense_top_k=top_k)
    file_type = search_type_to_embeddings_type[type.value]

    query = raw_query
    search_model = await sync_to_async(get_default_search_model)()
    if max_distance is None:
        threshold = search_model.bi_encoder_confidence_threshold
        max_distance = threshold if threshold is not None else math.inf

    # Encode the query using the bi-encoder
    if question_embedding is None:
        with timer("Query Encode Time", logger, state.device):
            question_embedding = await sync_to_async(state.embeddings_model[search_model.name].embed_query)(
                retrieval_query_text(query)
            )

    # Find relevant entries for the query
    with timer("Search Time", logger, state.device):

        def fetch():
            return list(
                EntryAdapters.search_with_embeddings(
                    raw_query=raw_query,
                    embeddings=question_embedding,
                    max_results=top_k,
                    file_type_filter=file_type,
                    max_distance=max_distance,
                    user=user,
                    agent=agent,
                )
            )

        hits = await sync_to_async(fetch)()

    return hits


def to_retrieval_hits(hits: Iterable[DbEntry]) -> List[RetrievalHit]:
    """Convert distance-ordered dense rows without merging shared corpus IDs.

    Consume evaluated rows in async callers to avoid synchronous ORM I/O.
    The legacy query/collation path continues to use the original Entry rows.
    """
    return [
        RetrievalHit(entry_id=hit.pk, dense_rank=rank, cosine_distance=hit.distance)
        for rank, hit in enumerate(hits, start=1)
    ]


def collate_results(hits, dedupe=True):
    hit_ids = set()
    hit_hashes = set()
    for hit in hits:
        if dedupe and (hit.hashed_value in hit_hashes or hit.corpus_id in hit_ids):
            continue

        else:
            hit_hashes.add(hit.hashed_value)
            hit_ids.add(hit.corpus_id)
            yield SearchResponse.model_validate(
                {
                    "entry": hit.raw,
                    "score": hit.distance,
                    "corpus_id": str(hit.corpus_id),
                    "additional": {
                        "source": hit.file_source,
                        "file": hit.file_path,
                        "uri": hit.url,
                        "compiled": hit.compiled,
                        "heading": hit.heading,
                    },
                }
            )


def deduplicated_search_responses(hits: List[SearchResponse]):
    unique = {}
    for hit in hits:
        key = hit.additional["compiled"]
        if key not in unique:
            unique[key] = hit.model_copy(deep=True)
            unique[key].additional["queries"] = []
        query = hit.additional.get("query")
        if query is not None and query not in unique[key].additional["queries"]:
            unique[key].additional["queries"].append(query)
    yield from unique.values()


def rerank_and_sort_results(hits, query, rank_results, search_model_name, mode=RetrievalMode.VECTOR):
    # Rerank results if explicitly requested, if can use inference server
    # AND if we have more than one result
    rank_results = (rank_results or state.cross_encoder_model[search_model_name].inference_server_enabled()) and len(
        list(hits)
    ) > 1

    # Score all retrieved entries using the cross-encoder
    if rank_results:
        hits = cross_encoder_score(query, hits, search_model_name)

    # Sort results by cross-encoder score followed by bi-encoder score
    if mode == RetrievalMode.VECTOR:
        hits = sort_results(rank_results=rank_results, hits=hits)
    else:
        metric = "rrf_score" if mode == RetrievalMode.HYBRID else "bm25_score"
        hits.sort(key=lambda hit: (-hit.additional[metric], hit.additional["entry_id"]))
        if rank_results:
            hits.sort(key=lambda hit: hit.cross_score)

    return hits


def setup(
    text_to_entries: Type[TextToEntries],
    files: dict[str, str],
    regenerate: bool,
    user: KhojUser,
    config=None,
) -> Tuple[int, int]:
    if config:
        num_new_embeddings, num_deleted_embeddings = text_to_entries(config).process(
            files=files, user=user, regenerate=regenerate
        )
    else:
        num_new_embeddings, num_deleted_embeddings = text_to_entries().process(
            files=files, user=user, regenerate=regenerate
        )

    if files:
        file_names = [file_name for file_name in files]

        logger.info(
            f"Deleted {num_deleted_embeddings} entries. Created {num_new_embeddings} new entries for user {user} from files {file_names[:10]} ..."
        )

    return num_new_embeddings, num_deleted_embeddings


def cross_encoder_score(query: str, hits: List[SearchResponse], search_model_name: str) -> List[SearchResponse]:
    """Score all retrieved entries using the cross-encoder"""
    try:
        with timer("Cross-Encoder Predict Time", logger, state.device, log_level=logging.INFO):
            cross_scores = state.cross_encoder_model[search_model_name].predict(query, hits)
    except requests.exceptions.HTTPError as e:
        logger.error(f"Failed to rerank documents using the inference endpoint. Error: {e}.", exc_info=True)
        cross_scores = [0.0] * len(hits)

    # Convert cross-encoder scores to distances and pass in hits for reranking
    for idx in range(len(cross_scores)):
        hits[idx]["cross_score"] = 1 - cross_scores[idx]

    return hits


def sort_results(rank_results: bool, hits: List[dict]) -> List[dict]:
    """Order results by cross-encoder score followed by bi-encoder score"""
    with timer("Rank Time", logger, state.device):
        hits.sort(key=lambda x: x["score"])  # sort by bi-encoder score
        if rank_results:
            hits.sort(key=lambda x: x["cross_score"])  # sort by cross-encoder score
    return hits


def hydrate_retrieval_hits(hits, raw_query, user, type, embedding, agent):
    """Reapply scope when fetching payloads, without a Dense distance cutoff."""
    if not hits:
        return []
    return list(
        EntryAdapters.apply_filters(user, raw_query, search_type_to_embeddings_type[type.value], agent)
        .filter(pk__in=[hit.entry_id for hit in hits])
        .annotate(distance=CosineDistance("embeddings", embedding))
    )


def collate_retrieval_hits(hits, rows):
    from khoj.search_type.hybrid_search import RetrievalUnavailable

    by_id = {row.pk: row for row in rows}
    results = []
    for hit in hits:
        row = by_id.get(hit.entry_id)
        if row is None:  # Deleted or no longer in scope since recall.
            continue
        if row.distance is None or not math.isfinite(row.distance):
            raise RetrievalUnavailable("Candidate has no valid embedding cosine distance")
        response = next(collate_results([row], dedupe=False))
        response.additional.update(
            entry_id=hit.entry_id,
            dense_rank=hit.dense_rank,
            bm25_rank=hit.bm25_rank,
            bm25_score=hit.bm25_score,
            rrf_score=hit.rrf_score,
        )
        results.append(response)
    return results
