"""Seed CSV reading and type coercion.

CSV gives us strings. The casts declared on each spec in
:mod:`agrirag.graph.schema` turn them into the types the graph should store, so
that numeric comparisons in Cypher (``ph_min <= 7.0``) actually work.
"""

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agrirag.graph.schema import LIST_SEPARATOR, Cast

_TRUE = frozenset({"true", "yes", "1", "t", "y"})
_FALSE = frozenset({"false", "no", "0", "f", "n"})


class SeedDataError(ValueError):
    """Raised when a seed file cannot be parsed into the declared shape."""


def _coerce(value: str, cast: Cast, *, source: str, field: str) -> Any:
    try:
        if cast == "int":
            return int(value)
        if cast == "float":
            return float(value)
        if cast == "bool":
            lowered = value.strip().lower()
            if lowered in _TRUE:
                return True
            if lowered in _FALSE:
                return False
            raise ValueError(f"{value!r} is not a boolean")
        if cast == "list":
            return [part.strip() for part in value.split(LIST_SEPARATOR) if part.strip()]
    except ValueError as exc:
        raise SeedDataError(f"{source}: cannot cast {field}={value!r} to {cast}: {exc}") from exc
    raise SeedDataError(f"{source}: unknown cast {cast!r} for field {field}")


def read_rows(path: Path, casts: dict[str, Cast] | None = None) -> list[dict[str, Any]]:
    """Read a seed CSV into typed dicts.

    Empty cells are dropped rather than stored as empty strings, so a property
    that does not apply (``npk`` on a pesticide) is simply absent from the node.
    """
    if not path.exists():
        raise SeedDataError(f"seed file not found: {path}")

    casts = casts or {}
    rows: list[dict[str, Any]] = []

    with path.open(encoding="utf-8", newline="") as handle:
        for line_no, raw in enumerate(csv.DictReader(handle), start=2):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key is None:
                    raise SeedDataError(f"{path.name}:{line_no}: more columns than headers")
                if value is None or value.strip() == "":
                    continue
                cast = casts.get(key)
                row[key] = (
                    _coerce(value, cast, source=f"{path.name}:{line_no}", field=key)
                    if cast
                    else value.strip()
                )
            if row:
                rows.append(row)

    return rows


def iter_batches(rows: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    """Yield ``rows`` in chunks of at most ``size``."""
    for start in range(0, len(rows), size):
        yield rows[start : start + size]
