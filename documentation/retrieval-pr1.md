# Retrieval foundation (PR-1)

Dense retrieval keeps its existing Entry-row return format, collation, deduplication,
cosine-distance `SearchResponse.score`, and Cross-Encoder scoring and ordering.
`text_search.query(..., top_k=10)` accepts a positive integer at the end of the
existing argument list. `execute_search` requests `max(10, n)` candidates before
reranking and the existing final slice. Deduplication and filtering may reduce the
number of returned results. The search API accepts `n` from 1 to 1000 (422 outside
this range); the internal helper rejects invalid values with ValueError, and
`n=None` retains the default of 5. This bounds the newly configurable workload.

`RetrievalMode` and `RetrievalOptions` are internal contracts. VECTOR remains the
default; BM25 and HYBRID are reserved enum values, with no execution or public API
options. `to_retrieval_hits` converts evaluated, distance-ordered Entry rows to
1-based dense ranks using each integer Entry primary key. Shared corpus IDs are
not merged. Scores remain independent; missing retriever evidence stays None.

`EntryAdapters.apply_filters` scopes to a user, a previously authorized agent, or
the union of both. No scope returns no rows. Callers retain responsibility for
agent authorization; `execute_search` checks agent access when a user is present.
The agent-only path supports already-authorized chat/share callers. File type is
always applied; matching date rows cannot duplicate candidates. Search caches are
partitioned by user and agent. None uses the model threshold for direct queries;
0.0 is explicit, including a configured zero threshold. The API retains its existing
unbounded-distance default when max_distance is omitted.

BM25 can reuse these types and scoped queryset in PR-2. Its backend and conversion
remain to be added; fusion and alternate-mode dispatch are outside this change.
