---
sidebar_position: 1
---

# Optional BM25 index administration

`setup_bm25_index` builds an auxiliary ParadeDB index over existing `Entry`
chunks. The command does not dispatch retrieval, change Dense retrieval, or fuse
results. PR-2 also provides an independent BM25 adapter (see below). Existing text, embeddings, ownership and vector indexes are preserved.
No migration or server startup hook installs the extension or creates this index.
The default pgvector deployment remains supported without `pg_search`.

## Supported environment

This command targets PostgreSQL **15–18** and stable **pg_search 0.26.x**. Other
versions are reported and refused for creation until validated. Pin the extension
package version and PostgreSQL major version in your deployment rather than
following `latest`. These requirements are based on the
[0.26.0 build targets](https://github.com/paradedb/paradedb/blob/v0.26.0/pg_search/Cargo.toml),
[extension dependencies](https://github.com/paradedb/paradedb/blob/v0.26.0/pg_search/pg_search.control),
and current [index documentation](https://www.paradedb.com/docs/reference/indexing/create-index).

The extension must already be installed on the server **and in the selected
database**, with `vector` installed first. `pg_search` requires a compatible
server/OS/architecture build and must be in `shared_preload_libraries`, followed
by a PostgreSQL restart. See the [official installation guide](https://www.paradedb.com/docs/operate/deploy/self-hosted/extension)
and the [0.26.0 preload check](https://github.com/paradedb/paradedb/blob/v0.26.0/pg_search/src/lib.rs).
Have the DBA install the matching package, preserve existing preload entries,
arrange a restart, and explicitly run `CREATE EXTENSION vector` (if absent) and
`CREATE EXTENSION pg_search` in the intended database. The command does none of
these operations and does not require a ParadeDB Python client.

Creation requires effective ownership of the Entry table, schema USAGE/CREATE,
and visibility of preload settings. Ordinary roles cannot always read those
settings. Inspection reports unknown preload status rather than failing; creation
is then refused. Ask the DBA to inspect or grant `pg_read_all_settings` if
appropriate. Existing correctly defined indexes need no additional creation
permission to be recognized.

Do not swap the default PostgreSQL image or reuse its volume with an arbitrary
new server version. Plan backups, a restore rehearsal, extension/package
compatibility and an isolated staging test before any deployment change. No
volume deletion is required for this feature.

## Commands

From the repository root, using the existing virtual environment and the same
`POSTGRES_*` variables as Khoj:

```bash
export DJANGO_SETTINGS_MODULE=khoj.app.settings
export PYTHONPATH=src

# Default and explicit checks are read-only inventory operations.
.venv/bin/python -m django setup_bm25_index
.venv/bin/python -m django setup_bm25_index --check

# Print the intended SQL, target database and prerequisites without executing DDL.
.venv/bin/python -m django setup_bm25_index --dry-run

# Run only after reviewing the target and arranging a maintenance window.
.venv/bin/python -m django setup_bm25_index --create

# Verify the actual catalog state after creation.
.venv/bin/python -m django setup_bm25_index --check
```

All modes accept `--database default` (or another configured Django database
alias). An alias selects a configured connection, not an arbitrary database name;
checks, locking, DDL and postverification use that same connection. Verify the
reported database/user/table before creating anything.

Checks and previews return success when inspection completes, even if the optional
extension/index is missing; read the reported status and prerequisites. Creation
returns a nonzero exit status on prerequisite, conflict, lock, DDL, commit or
verification failure. A connection failure is also an error. Unhandled errors are
never treated as a successful build.

## Index definition

With the repository's actual model metadata and the usual `public` schema:

```sql
CREATE INDEX "khoj_entry_bm25_v1" ON "public"."database_entry" USING paradedb
("id", ("compiled"::"pdb"."icu"), "heading", "user_id", "agent_id",
 "file_type", "file_source", "file_path");
```

The command resolves the real table/schema and columns from Django metadata and
catalogs and quotes every identifier. `id` is the Entry primary key; `compiled`
is the same chunk used for embedding. ICU is used explicitly for the body;
heading retains the default Unicode tokenizer. Ownership/type/source/path fields
are available for subsequent filtering. Embeddings are not part of this index.
No deprecated `key_field`, untested `pnorms`, partial predicate or concurrency
option is added.

[ICU](https://www.paradedb.com/docs/reference/tokenizers/available-tokenizers/icu)
is intended for multilingual text. The integration test checks English, Chinese
and mixed text tokenization/retrieval as smoke tests; this is not a systematic
Chinese segmentation or retrieval-quality evaluation. Such evaluation belongs to
the retriever stage.

## Existing indexes, locking and failures

Only one ParadeDB index may exist per table. Inspection includes all Entry
indexes, both `paradedb` and legacy `bm25` access methods, and any same-name
relation in the target schema. A managed index is accepted only when its method,
table, ordered expressions, tokenizer, validity/readiness/liveness, absence of a
predicate, default options and non-unique structure match v1. Deparsed ICU type
names are resolved using the actual `pdb.icu` type under the session search path.
A different definition, another ParadeDB index, or an invalid/incomplete build
is reported as a conflict/error. No indexes are automatically dropped or repaired.
Ask the DBA to inspect the printed definitions and arrange any manual remediation
with backups and a maintenance plan.

This command uses ordinary `CREATE INDEX`, **not `CONCURRENTLY`**. Concurrent
build support has not been verified for this target. Reads can continue but
writes can be blocked. Build time, CPU, memory, disk and WAL usage depend on the
number and size of chunks; benchmark a representative staging copy first.

Defaults are a **5-second lock timeout** and **1800-second statement timeout**,
set locally inside the build transaction. Both accept positive integer seconds:

```bash
.venv/bin/python -m django setup_bm25_index --create \
  --lock-timeout 5 --statement-timeout 1800
```

The command refuses caller-managed transactions, serializes cooperating setup
commands with a transaction advisory lock, rechecks state before DDL, and checks
the resulting index before committing. Success is printed only after commit.
An unrelated concurrent DDL operation can still cause an error; inspect and retry
rather than assuming an index was installed. Failed builds roll back through
Django's transaction handling; inspect PostgreSQL logs/catalogs if the server or
extension reports cleanup problems. Settings do not leak into subsequent work.

Useful read-only DBA checks:

```sql
SHOW server_version;
SHOW shared_preload_libraries;
SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pg_search');
SELECT name, default_version, installed_version FROM pg_available_extensions
WHERE name IN ('vector', 'pg_search');
SELECT schemaname, indexname, indexdef FROM pg_indexes WHERE tablename = 'database_entry';
SELECT pid, phase, blocks_done, blocks_total FROM pg_stat_progress_create_index;
```

For a permission error, inspect the effective role and schema privileges. For a
missing extension, have the DBA install/enable the matched build. For an invalid
index, inspect `pg_index.indisvalid`, `indisready` and `indislive` and any ongoing
build before manual action. For timeouts, review locks, workload and maintenance
window; do not bypass them with an unlimited timeout blindly.

## Real query and execution-plan verification

On a disposable/staging database containing the supported extension and index:

```sql
SELECT id, pdb.score(id) AS bm25_score
FROM public.database_entry
WHERE compiled ||| 'keyword retrieval'
ORDER BY bm25_score DESC, id
LIMIT 5;

EXPLAIN (ANALYZE, BUFFERS)
SELECT id, pdb.score(id) AS bm25_score
FROM public.database_entry
WHERE compiled ||| 'keyword retrieval'
ORDER BY bm25_score DESC, id
LIMIT 5;

SELECT 'Hello world! 你好!'::pdb.icu::text[];
```

Use the schema/table reported by the command. Scores are genuine
[BM25 scores](https://www.paradedb.com/docs/reference/full-text/score), with larger
values ranking first. Application queries bind user text as parameters
and apply the existing user/authorized-Agent scope; the index itself grants no
row-level authorization. It does not replace `EntryAdapters.apply_filters`.

Record the actual access path/index, buffers, execution time, filters and sorts.
An existing index alone does not prove the query uses it. A small table can lead
to a different valid plan; use representative independent data when assessing
performance rather than forcing planner settings or claiming a benchmark.

## Tests and current verification limits

Ordinary PostgreSQL development environments can run:

```bash
.venv/bin/pytest -q tests/test_bm25_index_command.py tests/test_bm25_index_integration.py
```

The lifecycle integration test is explicitly marked `bm25_integration` and
skipped unless `KHOJ_TEST_BM25=1`. When opted in, missing/unsupported prerequisites
are failures rather than silent skips. It never installs extensions or drops
existing indexes.

Use a **fresh disposable `test_` database**, never a production database. For
example, when the app DB setting is `POSTGRES_DB=khoj`, pytest uses `test_khoj`
unless overridden. Have the DBA provision that isolated test database, install
vector/pg_search 0.26.x there, and configure preloading before running pytest
with `--reuse-db`; Django applies the existing migrations. Use a separate cluster
or suitable test template if provisioning through CI. Do not point Django's TEST
configuration at a production database. The test refuses a preexisting managed
index: use a fresh disposable database instead of deleting indexes to rerun it.

```bash
KHOJ_TEST_BM25=1 .venv/bin/pytest --reuse-db -m bm25_integration \
  tests/test_bm25_index_integration.py -s --junitxml=bm25-integration.xml
```

The lifecycle test creates the real index, checks initial data preservation,
queries BM25 scores, records JSON EXPLAIN/timing in JUnit properties, compares the
index OID after repeated creation, and checks insert/update/delete and rollback
visibility. It also prints the plan for inspection and records whether a
ParadeDB/index access path was observed. No runtime success or performance is
claimed unless that test actually runs on the supported extension.

The original task 2.1 run used PostgreSQL 16.2/vector 0.6.2 without pg_search.
The subsequent PR-2 verification on 2026-10-29 used PostgreSQL **15.19**,
vector **0.8.7**, pg_search **0.26.3** and an isolated database named
`test_khoj_pr2_retriever_20261029`. The existing application database and its
BM25 index were inspected read-only, never rebuilt or used for lifecycle writes.


## Disabling

PR-2 adds no BM25 dispatch or feature toggle, so existing search remains Dense.
To stop index builds, do not invoke `--create`. When retrieval dispatch is added,
disable that dispatch first; leaving an unused index in place is safe for existing
query behavior but still incurs storage/write overhead. Any later index removal
or extension/package change is a separate DBA-reviewed maintenance operation.
The command does not remove data, vector indexes, extensions, volumes or backups.

## Independent BM25 adapter (PR-2 task 2.2)

```python
from khoj.database.adapters import EntryAdapters

hits = EntryAdapters.search_with_bm25(
    raw_query='ParadeDB BM25 file:"notes/*.pdf" -"obsolete"',
    user=user,
    max_results=40,
    file_type_filter="pdf",
    agent=None,
    file_source_filter="computer",
)
# Async code:
# hits = await sync_to_async(EntryAdapters.search_with_bm25)("BM25", user)
```

The synchronous interface returns `list[RetrievalHit]` using the existing
`khoj.search_type.retrieval_types` definition. Each result contains its actual
`Entry.pk`, `bm25_rank` (1-based), and native floating-point `bm25_score`.
Larger scores rank first, ties use `Entry.pk ASC`. Independent chunks sharing a
`corpus_id` remain independent candidates. `dense_rank`/`cosine_distance` stay
`None`; `rrf_score` stays at its default `0.0`. No BM25 score is put into
`SearchResponse.score`.

`max_results` defaults to 40 and can be any nonnegative Python integer, excluding
bool. Zero returns an empty list. Negative/noninteger limits or a non-string
query raise `ValueError`. Blank queries and queries containing only recognized
Khoj filters return an empty list without executing a search. Punctuation-only
input is tokenized by ParadeDB; it does not become an all-documents query.

The adapter passes the **original query** to `EntryAdapters.apply_filters`.
That existing QuerySet applies `(user=user OR agent=agent)` ownership, or no
access when both are absent; required/excluded raw-text words, glob file-path
inclusion/exclusion, date joins (with DISTINCT), and optional file type. Source
is an additional exact `file_source_filter` applied to the same QuerySet; there
is no new source search syntax. File type is the existing Entry type value
(e.g. `pdf`), not a new SearchType enum.

Before adding an Agent scope, BM25 checks `AgentAdapters.is_agent_accessible`
against the **current database row**. The existing asynchronous accessibility
method delegates to this same policy: public/protected Agents or the creator
are allowed; other private Agents raise `PermissionError` in BM25. Agent row
absence raises `Agent.DoesNotExist`. Public/protected access is existing Khoj
policy, not a BM25 permission expansion. An allowed Agent adds its copied
entries to the user's entries; it does not add the creator's entire library.

For matching, recognized date/word/file syntax is removed from the query body;
excluded-file stripping and leading date syntax are handled locally because
legacy defilter helpers leave these edge cases. Dense parsing is unchanged.
`lexical_search.py` uses a small Django boolean expression and `Func`, without
an additional dependency or globally registered lookup. The official
[Django integration](https://github.com/paradedb/django-paradedb) was considered;
local native expressions keep existing QuerySet filters and need only the two
operators used here. Raw SQL with copied authorization predicates would create
a second permission implementation and was therefore avoided.

The generated query has this shape (all text and filter values are bound):

```sql
SELECT id, pdb.score(id) AS bm25_score
FROM database_entry
WHERE (user_id = %s OR agent_id = %s)
  AND ...existing filters...
  AND compiled ||| %s
ORDER BY bm25_score DESC, id ASC
LIMIT 40;
```

Django inserts the validated integer slice into LIMIT; it never interpolates
user query text into SQL. `|||` is a tokenized
[match disjunction](https://www.paradedb.com/docs/reference/full-text/match),
not a query-language parser; quotes and punctuation are literal search input.
Filtering precedes truncation inside SQL, with no Python matching/filtering,
all-hit materialization, or per-candidate lookup. Only PK and score are fetched.
Missing extension/operator/schema, missing/invalid index and query/connection
failures propagate as Django database exceptions. There is no automatic Dense
fallback, fake score, extension installation or index repair. Use the existing
administrative `--check` for detailed prerequisite/index diagnostics.

## Entry writes and index consistency (PR-2 task 2.3)

Entry rows remain the source of truth. The normal PostgreSQL/ParadeDB index
maintains inserts, updates and deletes transactionally. Verified successful
writes require **no refresh, dual write, queue or manual REINDEX**.

| Operation | Khoj path | Verification |
| --- | --- | --- |
| Insert / bulk insert | `TextToEntries.update_embeddings`, `Entry.objects.bulk_create` | New lexemes/PKs immediately retrieved; real ingestion tested |
| Update / bulk update | `QuerySet.update`, `bulk_update`; `change_default_model.py` also uses bulk update | Changed body loses old terms and gains new terms |
| Delete / batch delete | `EntryAdapters.delete_entry_by_file`, `delete_entry_by_hash`, `delete_all_entries`; async equivalents use the same table | Deleted chunks absent; batch size 1 exercises multiple batches |
| File content replacement | `update_embeddings`: hash new compiled chunks, bulk insert, delete obsolete hashes | Old terms/PKs gone, new terms present, unchanged repeat does not duplicate |
| Regenerate | `update_embeddings(regenerate=True)` clears user's file type then inserts | New PKs, no residual old rows, same final content searchable |
| Chunk split | `split_entries_by_max_tokens` creates a corpus UUID per original entry | Shared corpus UUID yields multiple distinct retrievable PKs |
| Agent copy/update/delete | `AgentAdapters.atomic_update_agent` deletes copies and bulk copies creator's selected files; Agent deletion cascades | Copies have new PKs, file removal removes access, deletion preserves source rows |
| Commit | `transaction.atomic` | Insert/update/delete visible in an independent database session after commit |
| Rollback | `transaction.atomic` | Uncommitted changes visible locally, invisible to observer; rollback restores prior index and row state |

The file/online processors (plaintext, Markdown, org, PDF, GitHub, Notion, etc.)
feed the common embedding/write path. That pipeline is **not one atomic file
replacement**: regeneration deletes first, insertion is batched, and insertion
exceptions are logged by existing code. Agent file copies are snapshots, not
live links; editing a source does not refresh Agent copies until the Agent is
updated. PostgreSQL maintains the index of the rows actually committed; it does
not repair ingestion failures or change snapshot semantics. No ingestion behavior
was changed in this PR.

Review also found that the current `Entry.save()` override validates user/Agent
exclusivity but omits `super().save()`. Current ingestion uses `bulk_create` and
therefore writes successfully. Direct `Entry.objects.create()` / instance
`save()` should not be relied upon in this working tree; this preexisting model
issue is outside PR-2 and is not hidden by adding a BM25 synchronizer. INSERT
verification uses the actual bulk-insert path (including single-row inserts).

## Running the retriever/update tests

Standard PostgreSQL (without pg_search) supports the contract/mock SQL tests:

```bash
.venv/bin/pytest -q tests/test_bm25_retriever.py
```

Use the opt-in flag with the isolated, preconfigured test database described
above for real scores, multilingual recall, permission/filtering and all writers:

```bash
KHOJ_TEST_BM25=1 .venv/bin/pytest --reuse-db -q -s \
  tests/test_bm25_retriever_integration.py tests/test_bm25_index_updates.py \
  --junitxml=bm25-retriever.xml
```

New function-scoped fixtures validate an existing ready index or create it only
when missing in a `test_` database. They never drop indexes; explicit opt-in with
missing prerequisites/conflicting indexes fails. Without opt-in these real tests
are skipped, not considered verified. The no-extension unit test is skipped on
an extension-enabled server and is run separately on plain PostgreSQL.

The **task 2.1 lifecycle test requires an initially missing index**. The command
test `test_real_same_name_btree_is_reported_as_conflict` likewise requires no
preexisting managed index; run command tests on a separate plain PostgreSQL test
database. Run that
file first on a fresh test database, then the new files, or use separate test
databases. Repeatedly run only the new files to reuse their ready index. Do not
silence the lifecycle assertion or delete an application's index to accommodate
it. For example, the initial fresh-database run used this order:

```bash
KHOJ_TEST_BM25=1 .venv/bin/pytest --reuse-db -q -s \
  tests/test_bm25_retriever.py tests/test_bm25_index_integration.py \
  tests/test_bm25_retriever_integration.py tests/test_bm25_index_updates.py
```

First fresh-database run (including task 2.1 lifecycle): **36 passed, 1 skipped**
(the no-extension case), no failures. A separate initial plain PostgreSQL run
passed all 20 contract tests. Final verification expanded coverage to invalid
query types, missing index and tokenless input; see the delivery results below.

On the seven-row scoring fixture, the filtered query's JSON EXPLAIN observed
`Limit -> Custom Scan (ParadeDB Base Scan)`, index `khoj_entry_bm25_v1`,
`TopKScanExecState`, `pdb.score() desc, id asc`, limit 4, and the user ID inside
the Tantivy boolean query. Execution time was **0.286 ms**, shared hits **21**,
shared reads **0**, heap fetches **4**, no separate Sort node or temporary I/O.
The original lifecycle plan observed 0.328 ms, 22 shared hits and 0 reads on six
rows. Plans are printed and stored as JUnit properties. These tiny warm-cache
fixtures establish the query path, not performance/recall at production scale.
Regex/date filters may require extra PostgreSQL work; benchmark representative
corpora and inspect their plans before performance conclusions. Other server or
pg_search versions, concurrent ingestion workloads and crash recovery were not
validated in this run.

## Delivery verification and remaining limits

Actual local commands used two runners outside the source tree. `R` below means
`.venv/bin/python /tmp/khoj-pr2-retriever/run.py`, which sets existing Django
`POSTGRES_*` environment variables plus a temporary settings module selecting
`TEST.NAME=test_khoj_pr2_retriever_20261029`. `P` means
`XDG_RUNTIME_DIR=/tmp/khoj-deepseek .venv/bin/python /tmp/khoj-pr1/run_tests.py`,
the existing standalone PostgreSQL 16.2/vector-only test runner. Both run cached
embedding models offline. No source settings or deployment ports were changed.

| Invocation | Results | Real pg_search? |
| --- | --- | --- |
| Initial `KHOJ_TEST_BM25=1 R .venv/bin/pytest tests/test_bm25_retriever.py tests/test_bm25_index_integration.py tests/test_bm25_retriever_integration.py tests/test_bm25_index_updates.py -q -s` | 36 passed, 0 failed, 1 skipped | Yes; includes existing lifecycle |
| Final `KHOJ_TEST_BM25=1 R .venv/bin/pytest tests/test_bm25_retriever.py tests/test_bm25_retriever_integration.py tests/test_bm25_index_updates.py -q` | 41 passed, 0 failed, 1 skipped | Yes; contract mocks plus genuine integration |
| `P .venv/bin/pytest tests/test_bm25_retriever.py tests/test_bm25_index_command.py -q` | 74 passed, 0 failed, 0 skipped | No; verifies real missing-extension failure |
| `R .venv/bin/pytest tests/test_text_search.py tests/test_retrieval.py tests/test_multiple_users.py tests/test_bm25_index_command.py tests/test_agents.py -q` | 124 passed, 1 failed, 2 skipped, 8 fixture errors | Yes; Dense regressions still use pgvector |
| `P timeout --signal=INT --kill-after=30s 300 .venv/bin/pytest -q --tb=short --maxfail=1` | 261 passed, 1 failed, 3 skipped; stopped on first failure | No; opt-in BM25 tests not reached |

The mixed regression run's failure was the task 2.1 same-name B-tree test
colliding with the already-created BM25 index. Its subsequent separate plain-DB
run passed. All `test_text_search` cases ran (11 passed, 1 preexisting skip),
`test_retrieval` (59 passed), and `test_multiple_users` (4 passed).
The eight Agent fixture errors were missing chat API credentials:
`AiModelApi.api_key` received NULL before those test bodies could execute.
The two credential-independent Agent tests passed; the new real Agent copy and
permission tests also passed. These eight existing cases remain unverified:

- `test_create_or_update_agent_with_knowledge_base`
- `test_create_or_update_agent_with_knowledge_base_and_search`
- `test_agent_with_knowledge_base_and_search_not_creator`
- `test_agent_with_knowledge_base_and_search_not_creator_and_private`
- `test_agent_with_knowledge_base_and_search_not_creator_and_private_accessible_to_none`
- `test_multiple_agents_with_knowledge_base_and_users`
- `test_large_knowledge_base_atomic_update`
- `test_concurrent_agent_updates_atomicity`

The full-suite attempt stopped at `tests/test_helpers.py::test_reading_webpage`
with a network `TimeoutError`. It is not a full-green suite. Missing credentials,
external network dependency, production-scale performance, concurrency stress
and other pg_search versions remain verification limits. Existing model save,
non-atomic ingestion and Agent snapshots described above were not changed.

Independent review found no actionable regression. Ruff checks passed on all new
Python files and the adapter; `git diff --check` passed. Existing unrelated
uncommitted changes were preserved. No commit, merge, Docker image/volume change,
application-table lifecycle write, extension installation in the application
DB or existing-index recreation was performed.

## PR-3 reuse

Call `EntryAdapters.search_with_bm25(raw_query, user, max_results=40, agent=agent)`
as an independent branch (wrap with `sync_to_async` from async code). Merge by
`hit.entry_id` with the existing Dense candidates in PR-3. Interpret BM25's raw
score only within its own branch; consume `bm25_rank` for future RRF. Public
`/search`, Dense dispatch/defaults and Cross-Encoder behavior remain unchanged.
No hybrid mode or RRF is implemented here.
