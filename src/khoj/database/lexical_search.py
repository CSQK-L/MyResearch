"""Native ParadeDB expressions; ownership and Khoj filters belong to adapters."""

from django.db.models import BooleanField, FloatField, Func, QuerySet, Value

from khoj.database.models import Entry


class MatchAny(Func):
    """Tokenize bound text with the indexed column's tokenizer (no query parser)."""

    template = "(%(expressions)s)"
    arg_joiner = " ||| "
    arity = 2
    output_field = BooleanField()


def ranked_bm25_entries(entries: QuerySet[Entry], query: str, max_results: int) -> QuerySet[Entry]:
    """Keep all filtering in SQL before deterministic ranking and truncation.

    Missing pg_search, missing/invalid indexes and other database failures
    propagate as Django database exceptions. Never substitute another scorer.
    """
    return (
        entries.filter(MatchAny("compiled", Value(query)))
        .annotate(bm25_score=Func("pk", function="pdb.score", output_field=FloatField()))
        .order_by("-bm25_score", "pk")[:max_results]
    )
