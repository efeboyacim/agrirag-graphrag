"""Golden dataset loading.

Goldens live in YAML next to the seed data they were written against. Splitting
them by category keeps each file readable and makes the distribution visible at a
glance - it is otherwise very easy to end up with thirty happy-path questions and
no case where declining to answer is correct.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

GOLDENS_DIR = Path(__file__).parent / "goldens"

Route = Literal["graph", "semantic", "both", "any"]


class GoldenError(ValueError):
    """Raised when a golden file is malformed."""


@dataclass(frozen=True)
class Golden:
    """One evaluation case."""

    id: str
    input: str
    expected_output: str
    #: The route the router ought to pick. "any" for cases where it does not matter.
    expected_route: Route = "any"
    #: Routes that are also acceptable. A "both" question whose facts happen to be
    #: stated in prose can legitimately be answered by semantic search alone, and
    #: scoring that as a routing failure would punish a correct answer.
    acceptable_routes: tuple[str, ...] = ()
    #: Node ids the traversal should touch. Drives GraphPathRecall.
    expected_entity_ids: tuple[str, ...] = ()
    #: Ground-truth facts, consumed by HallucinationMetric (which needs `context`,
    #: not `retrieval_context` - the distinction is the whole point of that metric).
    context: tuple[str, ...] = ()
    #: True when the correct behaviour is to decline.
    must_abstain: bool = False
    hops: int = 0
    source_file: str = ""

    @property
    def category(self) -> str:
        return Path(self.source_file).stem or "unknown"

    def route_is_acceptable(self, actual: str) -> bool:
        """Is ``actual`` an acceptable route for this golden?

        A ``graph`` golden also accepts ``both``. The failure this metric exists
        to catch is the router *skipping* the graph on a question that needs a
        traversal; also consulting the corpus is a cost, not an error, and often
        produces a better-explained answer. Marking it wrong would push the
        router toward a narrowness that helps no one.

        Semantic goldens are not treated the same way: running a graph traversal
        on a definitional question is wasted work and usually means the router
        misread the question.
        """
        if self.expected_route == "any":
            return True
        if self.acceptable_routes:
            return actual in self.acceptable_routes
        if self.expected_route == "graph":
            return actual in {"graph", "both"}
        return actual == self.expected_route


def _parse(raw: dict[str, Any], source: str) -> Golden:
    missing = [k for k in ("id", "input", "expected_output") if not raw.get(k)]
    if missing:
        raise GoldenError(f"{source}: golden {raw.get('id', '?')} is missing {missing}")

    return Golden(
        id=raw["id"],
        input=raw["input"].strip(),
        expected_output=raw["expected_output"].strip(),
        expected_route=raw.get("expected_route", "any"),
        acceptable_routes=tuple(raw.get("acceptable_routes", ())),
        expected_entity_ids=tuple(raw.get("expected_entity_ids", ())),
        context=tuple(c.strip() for c in raw.get("context", ())),
        must_abstain=bool(raw.get("must_abstain", False)),
        hops=int(raw.get("hops", 0)),
        source_file=source,
    )


def load_goldens(directory: Path = GOLDENS_DIR) -> list[Golden]:
    """Load every golden, sorted by id for stable test ordering."""
    goldens: list[Golden] = []
    seen: set[str] = set()

    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        if not isinstance(raw, list):
            raise GoldenError(f"{path.name}: expected a list of goldens")

        for entry in raw:
            golden = _parse(entry, path.name)
            if golden.id in seen:
                raise GoldenError(f"duplicate golden id: {golden.id}")
            seen.add(golden.id)
            goldens.append(golden)

    if not goldens:
        raise GoldenError(f"no goldens found in {directory}")

    return sorted(goldens, key=lambda g: g.id)


@dataclass
class Distribution:
    """How the golden set is made up - reported so gaps are visible."""

    total: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_route: dict[str, int] = field(default_factory=dict)
    abstain_cases: int = 0
    multi_hop_cases: int = 0


def describe(goldens: list[Golden]) -> Distribution:
    """Summarise the golden set."""
    dist = Distribution(total=len(goldens))
    for golden in goldens:
        dist.by_category[golden.category] = dist.by_category.get(golden.category, 0) + 1
        dist.by_route[golden.expected_route] = dist.by_route.get(golden.expected_route, 0) + 1
        dist.abstain_cases += golden.must_abstain
        dist.multi_hop_cases += golden.hops >= 2
    return dist


#: Cap the number of goldens scored, for cheap CI runs. Unset means all of them.
SUBSET_ENV = "EVAL_SUBSET"


def _apply_subset(goldens: list[Golden]) -> list[Golden]:
    """Trim to ``EVAL_SUBSET`` cases, keeping the category mix balanced.

    Round-robin across categories rather than taking the first N, so a reduced
    CI run does not silently become "only graph questions".
    """
    raw = os.getenv(SUBSET_ENV)
    if not raw:
        return goldens

    limit = int(raw)
    if limit >= len(goldens):
        return goldens

    by_category: dict[str, list[Golden]] = {}
    for golden in goldens:
        by_category.setdefault(golden.category, []).append(golden)

    picked: list[Golden] = []
    while len(picked) < limit:
        added = False
        for bucket in by_category.values():
            if bucket and len(picked) < limit:
                picked.append(bucket.pop(0))
                added = True
        if not added:
            break

    return sorted(picked, key=lambda g: g.id)


def active_goldens(directory: Path = GOLDENS_DIR) -> list[Golden]:
    """The goldens this run scores, after applying ``EVAL_SUBSET``.

    One definition, used by the agent sweep, the report and the test suites, so
    all three operate on exactly the same set. If the sweep ran a subset and the
    report scored the full set, the report would fail on missing runs; if the
    suites used the full set, per-case assertions would fail for goldens nobody
    ran.

    :func:`load_goldens` stays unsubsetted - it is the source of truth, and the
    golden-set integrity check in CI needs to see all of them.
    """
    return _apply_subset(load_goldens(directory))
