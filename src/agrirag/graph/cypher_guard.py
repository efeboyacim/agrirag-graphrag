"""Read-only validation for Cypher before it reaches the database.

This is the first of three independent defences against a generated query doing
something it should not:

1. **This guard** - rejects write clauses and injects a LIMIT.
2. **Session access mode** - :class:`~agrirag.graph.neo4j_store.Neo4jGraphStore`
   opens READ sessions, so the server itself refuses a write.
3. **Architecture** - the write path lives in :mod:`agrirag.ingestion.writer`,
   which the agent never imports.

Any one of them is sufficient. Three exist because the query text is partly
LLM-generated, and a prompt is not a security boundary.
"""

import re
from dataclasses import dataclass

DEFAULT_LIMIT = 50
MAX_LIMIT = 200

#: Clauses that mutate data or schema. Matched as whole words, case-insensitively.
FORBIDDEN_KEYWORDS: frozenset[str] = frozenset(
    {
        "create",
        "merge",
        "delete",
        "detach",
        "set",
        "remove",
        "drop",
        "foreach",
        "load",
        "using",  # USING PERIODIC COMMIT
    }
)

#: Procedure namespaces that can write, escape the sandbox, or read the filesystem.
FORBIDDEN_PROCEDURES: tuple[str, ...] = (
    "apoc.create",
    "apoc.merge",
    "apoc.refactor",
    "apoc.periodic",
    "apoc.load",
    "apoc.export",
    "apoc.import",
    "apoc.trigger",
    "apoc.cypher.run",
    "apoc.cypher.dorun",
    "dbms.",
    "db.create",
    "db.drop",
)

_LIMIT_RE = re.compile(r"\blimit\s+(\d+)\s*$", re.IGNORECASE)
_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_STRING_RE = re.compile(r"'[^']*'|\"[^\"]*\"")


class UnsafeCypherError(ValueError):
    """Raised when a query is rejected."""


@dataclass(frozen=True)
class GuardedQuery:
    """A query that passed validation."""

    cypher: str
    limit: int
    #: True when the guard added a LIMIT the original query lacked.
    limit_injected: bool


def _strip_noise(cypher: str) -> str:
    """Remove comments and string literals before keyword scanning.

    Without this, a query legitimately returning a property whose *value*
    contains the word "create" would be rejected, and - more importantly - a
    keyword hidden inside a comment would be missed.
    """
    return _STRING_RE.sub("''", _COMMENT_RE.sub(" ", cypher))


def validate(cypher: str, *, default_limit: int = DEFAULT_LIMIT) -> GuardedQuery:
    """Validate a read-only Cypher query and guarantee it carries a LIMIT.

    Raises :class:`UnsafeCypherError` if the query writes, calls a forbidden
    procedure, or contains multiple statements.
    """
    if not cypher or not cypher.strip():
        raise UnsafeCypherError("empty query")

    cleaned = cypher.strip().rstrip(";").strip()
    scannable = _strip_noise(cleaned)

    # Multiple statements: one query per call, so a write cannot be smuggled in
    # behind a legitimate read.
    if ";" in scannable:
        raise UnsafeCypherError("multiple statements are not allowed")

    lowered = scannable.lower()

    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", lowered):
            raise UnsafeCypherError(f"write clause '{keyword.upper()}' is not allowed")

    for procedure in FORBIDDEN_PROCEDURES:
        if procedure in lowered:
            raise UnsafeCypherError(f"procedure '{procedure}' is not allowed")

    # A read query must actually read something.
    if not re.search(r"\b(match|call|unwind|return|with)\b", lowered):
        raise UnsafeCypherError("query contains no readable clause")

    match = _LIMIT_RE.search(cleaned)
    if match:
        limit = int(match.group(1))
        if limit > MAX_LIMIT:
            # Clamp rather than reject: an over-large LIMIT is a judgement error,
            # not an attack, and failing the whole query helps nobody.
            cleaned = _LIMIT_RE.sub(f"LIMIT {MAX_LIMIT}", cleaned)
            limit = MAX_LIMIT
        return GuardedQuery(cypher=cleaned, limit=limit, limit_injected=False)

    return GuardedQuery(
        cypher=f"{cleaned}\nLIMIT {default_limit}",
        limit=default_limit,
        limit_injected=True,
    )


def is_safe(cypher: str) -> bool:
    """Convenience predicate for tests and logging."""
    try:
        validate(cypher)
    except UnsafeCypherError:
        return False
    return True
