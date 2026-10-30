"""Opt-in Entry BM25 index administration; never installed by migrations/startup."""

import re
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Optional

from django.core.management.base import BaseCommand, CommandError
from django.db import DatabaseError, connections, transaction
from django.db.backends.base.base import BaseDatabaseWrapper
from django.utils.connection import ConnectionDoesNotExist

from khoj.database.models import Entry

INDEX_NAME = "khoj_entry_bm25_v1"
INDEX_FIELDS = ("compiled", "heading", "user", "agent", "file_type", "file_source", "file_path")


@dataclass(frozen=True)
class IndexInfo:
    name: str
    table_oid: Optional[int]
    method: Optional[str]
    valid: bool
    ready: bool
    live: bool
    expressions: tuple[str, ...]
    predicate: Optional[str]
    options: tuple[str, ...]
    unique: bool
    definition: Optional[str]


@dataclass(frozen=True)
class Environment:
    database: str
    user: str
    server_version: str
    server_version_num: int
    preload: Optional[str]
    extensions: dict[str, str]
    available_extensions: dict[str, str]
    table_oid: Optional[int]
    schema: str
    table: str
    table_kind: Optional[str]
    can_create: bool
    access_method: bool
    icu_type: bool
    indexes: tuple[IndexInfo, ...]
    icu_type_name: str = "pdb.icu"


def quote_identifier(identifier: str) -> str:
    # PostgreSQL delimited identifiers escape embedded quotes by doubling them.
    # Django's quote_name() wraps identifiers but does not escape embedded quotes.
    return '"' + identifier.replace('"', '""') + '"'


def index_expressions(quote: bool = False) -> tuple[str, ...]:
    """Resolve columns from Django metadata, including FK column names."""
    columns = [Entry._meta.pk.column, *(Entry._meta.get_field(field).column for field in INDEX_FIELDS)]
    if quote:
        columns = [quote_identifier(column) for column in columns]
        columns[1] = f'({columns[1]}::"pdb"."icu")'
    else:
        columns[1] += "::pdb.icu"
    return tuple(columns)


def create_index_sql(schema: str, table: str) -> str:
    """Quote every identifier; schema/table originate from PostgreSQL catalogs."""
    quote = quote_identifier
    columns = ", ".join(index_expressions(quote=True))
    return f"CREATE INDEX {quote(INDEX_NAME)} ON {quote(schema)}.{quote(table)} USING paradedb ({columns});"


def inspect_database(connection: BaseDatabaseWrapper) -> Environment:
    """Read catalogs only, on the selected connection, even without pg_search."""
    if connection.vendor != "postgresql":
        raise CommandError("BM25 index inspection requires PostgreSQL; no database objects were changed.")
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT current_database(), current_user, current_setting('server_version'), "
            "current_setting('server_version_num')::int, "
            "(SELECT setting FROM pg_settings WHERE name='shared_preload_libraries'), current_schema()"
        )
        database, user, version, version_num, preload, default_schema = cursor.fetchone()
        cursor.execute("SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pg_search')")
        extensions = dict(cursor.fetchall())
        cursor.execute(
            "SELECT name, default_version FROM pg_available_extensions WHERE name IN ('vector', 'pg_search')"
        )
        available = dict(cursor.fetchall())
        cursor.execute(
            "SELECT c.oid, n.nspname, c.relname, c.relkind, "
            "pg_has_role(c.relowner, 'USAGE') AND has_schema_privilege(n.oid, 'CREATE') "
            "AND has_schema_privilege(n.oid, 'USAGE') "
            "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.oid=to_regclass(%s)",
            [quote_identifier(Entry._meta.db_table)],
        )
        table = cursor.fetchone()
        oid, schema, name, kind, can_create = table or (
            None,
            default_schema or "public",
            Entry._meta.db_table,
            None,
            False,
        )
        cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_am WHERE amname='paradedb'), "
            "to_regtype('pdb.icu') IS NOT NULL, "
            "CASE WHEN to_regtype('pdb.icu') IS NOT NULL THEN format_type(to_regtype('pdb.icu'), NULL) "
            "ELSE 'pdb.icu' END"
        )
        access_method, icu_type, icu_type_name = cursor.fetchone()
        # Include same-name non-index objects or indexes on another table in the target namespace.
        cursor.execute(
            "SELECT c.relname, i.indrelid, am.amname, i.indisvalid, i.indisready, i.indislive, "
            "ARRAY(SELECT pg_get_indexdef(c.oid, k, false) FROM generate_series(1, i.indnatts) k), "
            "pg_get_expr(i.indpred, i.indrelid), c.reloptions, i.indisunique, "
            "CASE WHEN i.indexrelid IS NOT NULL THEN pg_get_indexdef(c.oid) END "
            "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "LEFT JOIN pg_index i ON i.indexrelid=c.oid LEFT JOIN pg_am am ON am.oid=c.relam "
            "WHERE i.indrelid=%s OR (n.nspname=%s AND c.relname=%s) ORDER BY c.relname",
            [oid, schema, INDEX_NAME],
        )
        indexes = tuple(
            IndexInfo(
                row[0],
                row[1],
                row[2],
                bool(row[3]),
                bool(row[4]),
                bool(row[5]),
                tuple(row[6]),
                row[7],
                tuple(row[8] or ()),
                bool(row[9]),
                row[10],
            )
            for row in cursor.fetchall()
        )
    return Environment(
        database,
        user,
        version,
        version_num,
        preload,
        extensions,
        available,
        oid,
        schema,
        name,
        kind,
        bool(can_create),
        access_method,
        icu_type,
        indexes,
        icu_type_name,
    )


def index_status(environment: Environment) -> tuple[str, str]:
    """Conservatively accept only the complete, unfiltered v1 definition."""
    other = [i.name for i in environment.indexes if i.name != INDEX_NAME and i.method in ("paradedb", "bm25")]
    if other:
        return "conflict", f"other ParadeDB indexes on the table: {', '.join(other)}"
    named = next((i for i in environment.indexes if i.name == INDEX_NAME), None)
    if named is None:
        return "missing", "no managed BM25 index"
    if named.table_oid != environment.table_oid or named.method != "paradedb":
        return "conflict", "same name belongs to a different relation or access method"
    if not (named.valid and named.ready and named.live):
        return "invalid", f"valid={named.valid}, ready={named.ready}, live={named.live}; inspect failed builds manually"

    # pg_get_indexdef may add identifier quotes/parentheses around a cast argument.
    def normalize(expression: str) -> tuple[str, ...]:
        tokens = re.findall(r'"(?:[^"]|"")*"|::|[A-Za-z_][A-Za-z0-9_]*|\S', expression)
        return tuple(
            token[1:-1] if re.fullmatch(r'"[a-z_][a-z0-9_]*"', token) else token
            for token in tokens
            if token not in ("(", ")")
        )

    expected = list(index_expressions())
    # format_type resolves the same pdb.icu OID under the current search_path;
    # an unrelated visible type named icu will not make pdb.icu unqualified.
    expected[1] = f"{Entry._meta.get_field('compiled').column}::{environment.icu_type_name}"
    if (
        tuple(map(normalize, named.expressions)) != tuple(map(normalize, expected))
        or named.predicate is not None
        or named.options
        or named.unique
    ):
        return "conflict", "same-name index definition differs from v1 (columns/tokenizer/predicate/options/uniqueness)"
    return "ready", "valid managed BM25 index"


def prerequisites(environment: Environment) -> list[str]:
    issues = []
    if not 150000 <= environment.server_version_num < 190000:
        issues.append(f"PostgreSQL {environment.server_version}: supported majors are 15–18")
    if "vector" not in environment.extensions:
        issues.append("vector extension is not installed in this database")
    version = environment.extensions.get("pg_search", "not installed")
    if not re.fullmatch(r"0\.26\.\d+", version):
        issues.append(f"pg_search {version}: install a matching server build and enable stable 0.26.x explicitly")
    if environment.preload is None:
        issues.append(
            "shared_preload_libraries is not visible; ask the DBA for pg_read_all_settings or a privileged inspection"
        )
    else:
        libraries = [
            part.strip().strip('"').rsplit("/", 1)[-1].removesuffix(".so") for part in environment.preload.split(",")
        ]
        if "pg_search" not in libraries:
            issues.append("shared_preload_libraries must contain pg_search; arrange a server restart with the DBA")
    if environment.table_oid is None:
        issues.append("Entry table is missing; apply existing Django migrations first")
    elif environment.table_kind != "r":
        issues.append("this command supports ordinary Entry tables only")
    if not environment.can_create:
        issues.append("insufficient permission: effective table ownership and schema USAGE/CREATE are required")
    if not environment.access_method:
        issues.append("paradedb index access method is unavailable")
    if not environment.icu_type:
        issues.append("pdb.icu tokenizer type is unavailable")
    return issues


class Command(BaseCommand):
    help = "Inspect (default), preview, or explicitly create the optional Entry BM25 index."
    requires_system_checks = []

    def add_arguments(self, parser: ArgumentParser) -> None:
        modes = parser.add_mutually_exclusive_group()
        modes.add_argument("--check", action="store_true", help="Read-only catalog inspection (default).")
        modes.add_argument("--dry-run", action="store_true", help="Print SQL and prerequisites without DDL.")
        modes.add_argument("--create", action="store_true", help="Explicitly build the index in a maintenance window.")
        parser.add_argument("--database", default="default", help="Django database alias.")
        parser.add_argument("--lock-timeout", type=int, default=5, help="Positive lock timeout in seconds (default 5).")
        parser.add_argument(
            "--statement-timeout", type=int, default=1800, help="Positive build timeout in seconds (default 1800)."
        )

    def report(self, environment: Environment) -> None:
        status, detail = index_status(environment)
        for label, value in (
            ("Database", environment.database),
            ("Database user", environment.user),
            ("PostgreSQL version", environment.server_version),
            ("pgvector extension status", environment.extensions.get("vector", "not installed")),
            ("pg_search extension status", "installed" if "pg_search" in environment.extensions else "not installed"),
            ("pg_search version", environment.extensions.get("pg_search", "none")),
            ("Available extensions", environment.available_extensions),
            ("shared_preload_libraries", environment.preload),
            ("Create permission", environment.can_create),
            ("BM25 index name", INDEX_NAME),
            ("BM25 index table", f"{environment.schema}.{environment.table}"),
            ("BM25 index status", f"{status}: {detail}"),
        ):
            self.stdout.write(f"{label}: {value}")
        for index in environment.indexes:
            self.stdout.write(f"Index {index.name}: valid={index.valid}, ready={index.ready}, live={index.live}")
            self.stdout.write(f"BM25 index definition / existing index: {index.definition or '(not an index)'}")
        if not environment.indexes:
            self.stdout.write("BM25 index definition: none")
        for issue in prerequisites(environment):
            self.stdout.write(f"Prerequisite: {issue}")

    def handle(self, *args: object, **options: Any) -> None:
        try:
            connection = connections[options["database"]]
            environment = inspect_database(connection)
            self.report(environment)
            if options["dry_run"]:
                self.stdout.write(
                    "Preview only. Requires PostgreSQL 15–18, pg_search 0.26.x, vector, preload and ownership."
                )
                self.stdout.write(create_index_sql(environment.schema, environment.table))
                return
            if not options["create"]:
                return
            for name in ("lock_timeout", "statement_timeout"):
                if type(options[name]) is not int or options[name] <= 0:
                    raise CommandError(f"{name} must be a positive integer number of seconds")
            status, detail = index_status(environment)
            if status in ("conflict", "invalid"):
                raise CommandError(
                    f"BM25 index {status}: {detail}. No indexes removed; ask the DBA to inspect/repair manually."
                )
            if status == "ready":
                self.stdout.write(self.style.SUCCESS("Index already exists with the expected definition; no changes."))
                return
            issues = prerequisites(environment)
            if issues:
                raise CommandError("Cannot create BM25 index: " + "; ".join(issues))
            if connection.in_atomic_block or not connection.get_autocommit():
                raise CommandError("Run --create outside any caller-managed transaction (autocommit required).")
            with transaction.atomic(using=connection.alias):
                with connection.cursor() as cursor:
                    cursor.execute("SELECT set_config('lock_timeout', %s, true)", [f"{options['lock_timeout']}s"])
                    cursor.execute(
                        "SELECT set_config('statement_timeout', %s, true)", [f"{options['statement_timeout']}s"]
                    )
                    cursor.execute(
                        "SELECT pg_try_advisory_xact_lock(hashtextextended(%s, 0))",
                        [f"khoj:bm25:{environment.table_oid}"],
                    )
                    if not cursor.fetchone()[0]:
                        raise CommandError("Another BM25 setup command is running; retry later.")
                    # The same lock as a normal CREATE INDEX prevents rename/drop while
                    # rechecking identity; it can block writers, so both timeouts are finite.
                    target = f"{quote_identifier(environment.schema)}.{quote_identifier(environment.table)}"
                    cursor.execute(f"LOCK TABLE {target} IN SHARE MODE")
                    current = inspect_database(connection)
                    if current.table_oid != environment.table_oid:
                        raise CommandError("Entry table identity changed during preflight; retry after DBA inspection.")
                    status, detail = index_status(current)
                    if status == "missing" and not prerequisites(current):
                        cursor.execute(create_index_sql(current.schema, current.table))
                    elif status != "ready":
                        raise CommandError(
                            f"BM25 index preflight changed: {status}: {detail}; {prerequisites(current)}"
                        )
                    final = inspect_database(connection)
                    if index_status(final)[0] != "ready":
                        raise CommandError("BM25 post-create verification failed; transaction rolled back.")
            # Success is reported only after the transaction committed, not merely after DDL returned.
            self.report(final)
            self.stdout.write(self.style.SUCCESS("BM25 index created successfully or verified after concurrent setup."))
        except (DatabaseError, ConnectionDoesNotExist) as exc:
            raise CommandError(f"BM25 index operation failed: {exc}") from exc
