"""Cypher guard.

The guard is one of three independent defences (guard, READ session mode,
architectural separation of the write path). It is tested hardest because it is
the one that sees LLM-generated text, and a prompt is not a security boundary.
"""

import pytest

from agrirag.graph.cypher_guard import (
    MAX_LIMIT,
    GuardedQuery,
    UnsafeCypherError,
    is_safe,
    validate,
)
from agrirag.graph.queries import TEMPLATES

READ_QUERY = "MATCH (c:Crop {id: $crop_id}) RETURN c.name AS name"


# --------------------------------------------------------------------------
# Rejection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cypher",
    [
        "CREATE (n:Crop {id: 'x'}) RETURN n",
        "MATCH (c:Crop) SET c.name = 'pwned' RETURN c",
        "MATCH (c:Crop) DELETE c",
        "MATCH (c:Crop) DETACH DELETE c",
        "MERGE (c:Crop {id: 'x'}) RETURN c",
        "MATCH (c:Crop) REMOVE c.name RETURN c",
        "DROP CONSTRAINT crop_id_unique",
        "MATCH (c:Crop) FOREACH (x IN [1] | SET c.n = x)",
        "LOAD CSV FROM 'file:///etc/passwd' AS line RETURN line",
        "USING PERIODIC COMMIT LOAD CSV FROM 'x' AS l RETURN l",
    ],
)
def test_write_clauses_are_rejected(cypher: str) -> None:
    with pytest.raises(UnsafeCypherError):
        validate(cypher)


@pytest.mark.parametrize(
    "cypher",
    [
        "CALL apoc.create.node(['X'], {}) YIELD node RETURN node",
        "CALL apoc.periodic.iterate('MATCH (n) RETURN n', 'DELETE n', {}) "
        "YIELD batches RETURN batches",
        "CALL apoc.load.json('http://evil/x') YIELD value RETURN value",
        "CALL apoc.export.csv.all('out.csv', {}) YIELD file RETURN file",
        "CALL dbms.listConfig() YIELD name RETURN name",
        "CALL apoc.cypher.run('CREATE (n)', {}) YIELD value RETURN value",
    ],
)
def test_dangerous_procedures_are_rejected(cypher: str) -> None:
    with pytest.raises(UnsafeCypherError):
        validate(cypher)


def test_statement_stacking_is_rejected() -> None:
    """A write must not ride along behind a legitimate read."""
    with pytest.raises(UnsafeCypherError, match="multiple statements"):
        validate("MATCH (c:Crop) RETURN c; CREATE (n:Evil) RETURN n")


def test_a_write_hidden_in_a_comment_is_still_caught() -> None:
    """Comments are stripped before scanning, not after - so this must not pass
    by hiding the keyword where a naive scanner would skip it."""
    with pytest.raises(UnsafeCypherError):
        validate("MATCH (c:Crop)\n// harmless\nDELETE c")


@pytest.mark.parametrize("cypher", ["", "   ", "\n"])
def test_empty_queries_are_rejected(cypher: str) -> None:
    with pytest.raises(UnsafeCypherError, match="empty"):
        validate(cypher)


def test_a_query_with_no_read_clause_is_rejected() -> None:
    with pytest.raises(UnsafeCypherError, match="no readable clause"):
        validate("RETURNING nonsense")


# --------------------------------------------------------------------------
# Acceptance
# --------------------------------------------------------------------------


def test_a_plain_read_passes() -> None:
    result = validate(READ_QUERY)

    assert isinstance(result, GuardedQuery)
    assert result.cypher.startswith("MATCH")


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (a)-[:GROWN_IN]->(b) RETURN a, b",
        "UNWIND [1,2,3] AS x RETURN x",
        "MATCH (c:Crop) WITH c ORDER BY c.name RETURN c.name",
        "CALL db.index.fulltext.queryNodes('entity_names', 'wheat') YIELD node RETURN node.id",
        "MATCH (c:Crop) WHERE EXISTS { MATCH (c)-[:GROWN_IN]->() } RETURN c",
    ],
)
def test_legitimate_read_shapes_pass(cypher: str) -> None:
    assert is_safe(cypher)


def test_a_string_literal_containing_a_keyword_is_not_a_write() -> None:
    """Rejecting this would break any query returning prose that says "create"."""
    assert is_safe("MATCH (p:Practice) WHERE p.description CONTAINS 'create' RETURN p")


# --------------------------------------------------------------------------
# LIMIT handling
# --------------------------------------------------------------------------


def test_a_missing_limit_is_injected() -> None:
    result = validate(READ_QUERY, default_limit=25)

    assert result.limit_injected is True
    assert result.cypher.rstrip().endswith("LIMIT 25")


def test_an_existing_limit_is_respected() -> None:
    result = validate(f"{READ_QUERY} LIMIT 7")

    assert result.limit_injected is False
    assert result.limit == 7
    assert result.cypher.count("LIMIT") == 1


def test_an_oversized_limit_is_clamped_rather_than_rejected() -> None:
    """An over-large LIMIT is a judgement error, not an attack. Failing the whole
    query would help nobody."""
    result = validate(f"{READ_QUERY} LIMIT 100000")

    assert result.limit == MAX_LIMIT
    assert f"LIMIT {MAX_LIMIT}" in result.cypher


def test_a_trailing_semicolon_is_tolerated() -> None:
    assert is_safe(f"{READ_QUERY};")


# --------------------------------------------------------------------------
# The committed templates must themselves survive the guard
# --------------------------------------------------------------------------


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t.name)
def test_every_committed_template_passes_the_guard(template) -> None:
    """A template that trips its own guard would fail only at request time."""
    assert is_safe(template.cypher)
