"""One mode resolver and query preprocessing policy for Search and Chat."""

import os
import re

from khoj.search_filter.date_filter import DateFilter
from khoj.search_filter.file_filter import FileFilter
from khoj.search_filter.word_filter import WordFilter
from khoj.search_type.retrieval_types import RetrievalMode


def resolve_retrieval_mode(mode=None) -> RetrievalMode:
    if mode is not None:
        return RetrievalMode(mode)
    configured = os.getenv("KHOJ_RETRIEVAL_MODE", "vector")
    try:
        return RetrievalMode(configured)
    except ValueError as error:
        raise ValueError("KHOJ_RETRIEVAL_MODE must be vector, bm25 or hybrid") from error


def retrieval_query_text(query: str) -> str:
    """Strip directives for inference only; original query still scopes SQL."""
    text = re.sub(FileFilter.excluded_file_filter_regex, "", " " + query)
    for query_filter in (DateFilter(), WordFilter(), FileFilter()):
        text = query_filter.defilter(text)
    return text.strip()


def retrieval_query_filters(query: str) -> str:
    patterns = (
        DateFilter.date_regex,
        WordFilter.required_regex,
        WordFilter.blocked_regex,
        FileFilter.excluded_file_filter_regex,
        FileFilter.file_filter_regex,
    )
    return " ".join(match.group(0).strip() for match in re.finditer("|".join(f"(?:{p})" for p in patterns), query))
