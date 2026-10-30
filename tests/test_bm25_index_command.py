"""Command contracts and real PostgreSQL catalog checks without requiring pg_search."""

from dataclasses import replace
from io import StringIO
from unittest.mock import patch

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import DatabaseError, connection
from django.test.utils import CaptureQueriesContext


def command_module():
    from khoj.database.management.commands import setup_bm25_index

    return setup_bm25_index


@pytest.fixture
def available():
    module = command_module()
    return module.Environment(
        database="test_khoj",
        user="owner",
        server_version="16.2",
        server_version_num=160002,
        preload="pg_search",
        extensions={"vector": "0.8.2", "pg_search": "0.26.0"},
        available_extensions={"vector": "0.8.2", "pg_search": "0.26.0"},
        table_oid=123,
        schema="public",
        table="database_entry",
        table_kind="r",
        can_create=True,
        access_method=True,
        icu_type=True,
        indexes=(),
    )


def index_info(**overrides):
    module = command_module()
    values = dict(
        name=module.INDEX_NAME,
        table_oid=123,
        method="paradedb",
        valid=True,
        ready=True,
        live=True,
        expressions=(
            "id",
            "compiled::pdb.icu",
            "heading",
            "user_id",
            "agent_id",
            "file_type",
            "file_source",
            "file_path",
        ),
        predicate=None,
        options=(),
        unique=False,
        definition="CREATE INDEX khoj_entry_bm25_v1 ON public.database_entry USING paradedb (...)",
    )
    values.update(overrides)
    return module.IndexInfo(**values)


def invoke(environment, *args, **options):
    module = command_module()
    output = StringIO()
    with (
        patch.object(module, "inspect_database", return_value=environment),
        patch.object(module, "connections") as connections,
    ):
        conn = connections.__getitem__.return_value
        conn.vendor = "postgresql"
        conn.alias = "default"
        conn.in_atomic_block = False
        conn.ops.quote_name = connection.ops.quote_name
        call_command("setup_bm25_index", *args, stdout=output, **options)
    return output.getvalue(), conn


@pytest.mark.parametrize("args", [(), ("--check",), ("--dry-run",)])
def test_read_only_modes_never_execute_sql(available, args):
    output, conn = invoke(available, *args)
    conn.cursor.assert_not_called()
    assert "missing" in output
    if "--dry-run" in args:
        assert 'CREATE INDEX "khoj_entry_bm25_v1"' in output
        assert "test_khoj" in output


def test_sql_is_metadata_derived_and_identifiers_are_quoted():
    module = command_module()
    from khoj.database.models import Entry

    sql = module.create_index_sql('schema";--', 'table"name')
    assert '"schema"";--"."table""name"' in sql
    assert '"compiled"::"pdb"."icu"' in sql
    assert "embeddings" not in sql
    assert "CONCURRENTLY" not in sql and "IF NOT EXISTS" not in sql
    assert "key_field" not in sql and "pnorms" not in sql
    assert Entry._meta.db_table == "database_entry"
    assert Entry._meta.get_field("user").column == "user_id"


def test_existing_correct_index_is_idempotent(available):
    output, conn = invoke(replace(available, indexes=(index_info(),)), "--create")
    assert "already" in output.lower()
    conn.cursor.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"method": "btree"},
        {"table_oid": 456},
        {"expressions": ("id", "raw")},
        {
            "expressions": (
                "id",
                "compiled::pdb.unicode",
                "heading",
                "user_id",
                "agent_id",
                "file_type",
                "file_source",
                "file_path",
            )
        },
        {"predicate": "user_id IS NOT NULL"},
        {"options": ("pnorms=true",)},
        {"unique": True},
        {"method": None, "table_oid": None},
    ],
)
def test_definition_conflict_is_not_accepted(available, changes):
    with pytest.raises(CommandError, match="conflict"):
        invoke(replace(available, indexes=(index_info(**changes),)), "--create")


@pytest.mark.parametrize("flag", ["valid", "ready", "live"])
def test_invalid_or_incomplete_index_is_rejected(available, flag):
    with pytest.raises(CommandError, match="invalid"):
        invoke(replace(available, indexes=(index_info(**{flag: False}),)), "--create")


@pytest.mark.parametrize("method", ["paradedb", "bm25"])
def test_other_paradedb_index_is_a_conflict(available, method):
    with pytest.raises(CommandError, match="conflict"):
        invoke(replace(available, indexes=(index_info(name="existing", method=method),)), "--create")


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"extensions": {"vector": "0.8.2"}}, "pg_search"),
        ({"extensions": {"pg_search": "0.26.0"}}, "vector"),
        ({"server_version_num": 140009, "server_version": "14.9"}, "PostgreSQL"),
        ({"server_version_num": 190000, "server_version": "19"}, "PostgreSQL"),
        ({"extensions": {"vector": "0.8.2", "pg_search": "0.25.0"}}, "0.26"),
        ({"extensions": {"vector": "0.8.2", "pg_search": "0.27.0"}}, "0.26"),
        ({"extensions": {"vector": "0.8.2", "pg_search": "0.26.0-dev"}}, "0.26"),
        ({"preload": "pg_stat_statements"}, "preload"),
        ({"can_create": False}, "permission"),
        ({"access_method": False}, "paradedb"),
        ({"icu_type": False}, "icu"),
        ({"table_oid": None}, "migrations"),
        ({"table_kind": "p"}, "ordinary"),
    ],
)
def test_create_refuses_missing_prerequisites_before_ddl(available, changes, expected):
    with pytest.raises(CommandError, match=expected):
        invoke(replace(available, **changes), "--create")


def test_database_connection_failure_is_nonzero():
    module = command_module()
    with patch.object(module, "inspect_database", side_effect=DatabaseError("connection unavailable")):
        with pytest.raises(CommandError, match="connection unavailable"):
            call_command("setup_bm25_index", stdout=StringIO())


def test_selected_database_alias_is_used(available):
    module = command_module()
    with (
        patch.object(module, "connections") as connections,
        patch.object(module, "inspect_database", return_value=available) as inspect,
    ):
        conn = connections.__getitem__.return_value
        conn.vendor = "postgresql"
        call_command("setup_bm25_index", "--database=secondary", stdout=StringIO())
        connections.__getitem__.assert_called_once_with("secondary")
        inspect.assert_called_once_with(conn)


@pytest.mark.parametrize("field", ["lock_timeout", "statement_timeout"])
@pytest.mark.parametrize("value", [0, -1])
def test_unbounded_timeouts_are_rejected(available, field, value):
    with pytest.raises(CommandError, match="positive"):
        invoke(available, "--create", **{field: value})


def test_postcreate_check_happens_before_success(available):
    module = command_module()
    ready = replace(available, indexes=(index_info(),))
    output = StringIO()
    with (
        patch.object(module, "inspect_database", side_effect=[available, available, ready]) as inspect,
        patch.object(module, "connections") as connections,
        patch.object(module.transaction, "atomic"),
    ):
        conn = connections.__getitem__.return_value
        conn.vendor, conn.alias, conn.in_atomic_block = "postgresql", "secondary", False
        conn.ops.quote_name = connection.ops.quote_name
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = (True,)
        call_command("setup_bm25_index", "--create", stdout=output)
        assert inspect.call_count == 3
        statements = [args[0] for args, kwargs in cursor.execute.call_args_list]
        assert sum(sql.startswith("CREATE INDEX") for sql in statements) == 1
        assert any("pg_try_advisory_xact_lock" in sql for sql in statements)
        assert "created" in output.getvalue().lower()
        module.transaction.atomic.assert_called_once_with(using="secondary")


def test_failed_postcreate_check_never_reports_success(available):
    module = command_module()
    output = StringIO()
    with (
        patch.object(module, "inspect_database", return_value=available),
        patch.object(module, "connections") as connections,
        patch.object(module.transaction, "atomic"),
    ):
        conn = connections.__getitem__.return_value
        conn.vendor, conn.alias, conn.in_atomic_block = "postgresql", "default", False
        conn.ops.quote_name = connection.ops.quote_name
        conn.cursor.return_value.__enter__.return_value.fetchone.return_value = (True,)
        with pytest.raises(CommandError, match="verification"):
            call_command("setup_bm25_index", "--create", stdout=output)
    assert "created successfully" not in output.getvalue().lower()


def test_real_postgres_check_and_dry_run_are_read_only():
    if connection.vendor != "postgresql":
        pytest.skip("PostgreSQL catalog test")
    module = command_module()
    with CaptureQueriesContext(connection) as queries:
        environment = module.inspect_database(connection)
        assert environment.table == "database_entry"
        assert environment.table_oid is not None
        for args in [(), ("--check",), ("--dry-run",)]:
            call_command("setup_bm25_index", *args, stdout=StringIO())
    assert all(q["sql"].lstrip().upper().startswith(("SELECT", "SHOW")) for q in queries)


def test_real_missing_extension_refuses_create_without_ddl():
    module = command_module()
    environment = module.inspect_database(connection)
    if "pg_search" in environment.extensions:
        pytest.skip("requires ordinary deployment without pg_search")
    with CaptureQueriesContext(connection) as queries:
        with pytest.raises(CommandError, match="pg_search"):
            call_command("setup_bm25_index", "--create", stdout=StringIO())
    assert not any("CREATE INDEX" in q["sql"] for q in queries)


def test_quoted_column_name_is_not_mistaken_for_icu_type(available):
    fake = index_info(
        expressions=(
            "id",
            '"compiled::pdb.icu"',
            "heading",
            "user_id",
            "agent_id",
            "file_type",
            "file_source",
            "file_path",
        )
    )
    with pytest.raises(CommandError, match="conflict"):
        invoke(replace(available, indexes=(fake,)), "--create")


def test_real_same_name_btree_is_reported_as_conflict():
    module = command_module()
    if connection.vendor != "postgresql":
        pytest.skip("PostgreSQL catalog test")
    with connection.cursor() as cursor:
        cursor.execute('CREATE INDEX "khoj_entry_bm25_v1" ON "database_entry" ("compiled")')
    environment = module.inspect_database(connection)
    assert module.index_status(environment)[0] == "conflict"
    with pytest.raises(CommandError, match="conflict"):
        call_command("setup_bm25_index", "--create", stdout=StringIO())


def test_search_path_visible_icu_is_recognized_by_catalog_type_name(available):
    env = replace(
        available,
        icu_type_name="icu",
        indexes=(
            index_info(
                expressions=(
                    "id",
                    "compiled::icu",
                    "heading",
                    "user_id",
                    "agent_id",
                    "file_type",
                    "file_source",
                    "file_path",
                )
            ),
        ),
    )
    assert command_module().index_status(env)[0] == "ready"


def test_unqualified_wrong_icu_type_is_a_conflict(available):
    fake = index_info(
        expressions=("id", "compiled::icu", "heading", "user_id", "agent_id", "file_type", "file_source", "file_path")
    )
    assert command_module().index_status(replace(available, indexes=(fake,)))[0] == "conflict"


def test_unknown_preload_is_reported_and_create_refused(available):
    output, conn = invoke(replace(available, preload=None), "--check")
    assert "pg_read_all_settings" in output
    with pytest.raises(CommandError, match="pg_read_all_settings"):
        invoke(replace(available, preload=None), "--create")


def test_real_catalog_inspection_as_unprivileged_role():
    if connection.vendor != "postgresql":
        pytest.skip("PostgreSQL catalog test")
    import uuid

    role = "bm25_inspection_" + uuid.uuid4().hex
    with connection.cursor() as cursor:
        cursor.execute("SELECT rolsuper FROM pg_roles WHERE rolname=current_user")
        if not cursor.fetchone()[0]:
            pytest.skip("requires a test superuser to create a temporary unprivileged role")
        cursor.execute(f'CREATE ROLE "{role}" NOSUPERUSER')
        cursor.execute(f'SET LOCAL ROLE "{role}"')
        try:
            environment = command_module().inspect_database(connection)
            assert environment.preload is None
            output = StringIO()
            call_command("setup_bm25_index", "--check", stdout=output)
            assert "pg_read_all_settings" in output.getvalue()
        finally:
            cursor.execute("RESET ROLE")


def test_replaced_table_is_not_indexed_after_preflight(available):
    module = command_module()
    replaced_table = replace(available, table_oid=456)
    ready = replace(replaced_table, indexes=(index_info(table_oid=456),))
    with (
        patch.object(module, "inspect_database", side_effect=[available, replaced_table, ready]),
        patch.object(module, "connections") as connections,
        patch.object(module.transaction, "atomic"),
    ):
        conn = connections.__getitem__.return_value
        conn.vendor, conn.alias, conn.in_atomic_block = "postgresql", "default", False
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = (True,)
        with pytest.raises(CommandError, match="identity changed"):
            call_command("setup_bm25_index", "--create", stdout=StringIO())
        assert not any(args[0].startswith("CREATE INDEX") for args, kwargs in cursor.execute.call_args_list)


@pytest.mark.parametrize("failure", ["ddl", "commit"])
def test_ddl_and_commit_failures_are_not_reported_as_success(available, failure):
    module = command_module()
    ready = replace(available, indexes=(index_info(),))
    output = StringIO()
    with (
        patch.object(module, "inspect_database", side_effect=[available, available, ready]),
        patch.object(module, "connections") as connections,
        patch.object(module.transaction, "atomic") as atomic,
    ):
        conn = connections.__getitem__.return_value
        conn.vendor, conn.alias, conn.in_atomic_block = "postgresql", "default", False
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = (True,)
        if failure == "commit":
            atomic.return_value.__exit__.side_effect = DatabaseError("commit failed")
        else:

            def execute(sql, *args):
                if sql.startswith("CREATE INDEX"):
                    raise DatabaseError("ddl failed")

            cursor.execute.side_effect = execute
        with pytest.raises(CommandError, match=f"{failure} failed"):
            call_command("setup_bm25_index", "--create", stdout=output)
    assert "created successfully" not in output.getvalue()
