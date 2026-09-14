"""Entity linking: question text -> canonical node ids.

This runs *before* any Cypher is selected or generated, and it is the single
biggest reliability win in the retrieval layer. The dominant text-to-Cypher
failure mode is not bad syntax - it is the model confidently filtering on
``{name: 'corn'}`` when the graph stores ``'Maize'``, which returns zero rows and
looks like a missing fact rather than a lookup miss.

Resolving mentions against the fulltext index first means the query is always
parameterised with ids that provably exist. Aliases carry the Turkish names
(``bugday``, ``misir``, ``sune``), so questions in either language resolve.
"""

import logging
import re
from dataclasses import dataclass

from agrirag.graph.schema import FULLTEXT_INDEX_NAME
from agrirag.graph.store import GraphStore

logger = logging.getLogger(__name__)

#: Below this Lucene score a match is more likely coincidental token overlap than
#: a real mention. Tuned against the seed data: real hits score well above it.
MIN_SCORE = 1.0

#: Words that produce spurious fulltext hits because they appear inside entity
#: names ("Cotton bollworm" would match a question about cotton).
_STOPWORDS = frozenset(
    {
        "what",
        "which",
        "when",
        "where",
        "how",
        "why",
        "who",
        "can",
        "should",
        "does",
        "do",
        "is",
        "are",
        "the",
        "a",
        "an",
        "for",
        "in",
        "on",
        "at",
        "to",
        "of",
        "and",
        "or",
        "my",
        "me",
        "i",
        "it",
        "any",
        "some",
        "there",
        "with",
        "from",
        "that",
        "this",
        "these",
        "those",
        "grow",
        "grown",
        "use",
        "used",
        "using",
        "best",
        "good",
        "control",
        "treat",
        # Category words, not entity names. They appear *inside* stored names
        # ("Sunn pest", "Cotton bollworm"), so without this a question like
        # "what pests affect tomatoes?" resolves to Sunn pest on a bare token
        # match and the traversal proceeds from an entity nobody mentioned.
        # Singularising candidate phrases made this worse, since "pests"
        # then matches too.
        "pest",
        "pests",
        "crop",
        "crops",
        "disease",
        "diseases",
        "soil",
        "soils",
        "region",
        "regions",
        "practice",
        "practices",
        "input",
        "inputs",
        "treatment",
        "treatments",
        "weed",
        "weeds",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ-]{2,}")


@dataclass(frozen=True)
class EntityRef:
    """A resolved graph entity."""

    id: str
    name: str
    label: str
    score: float
    #: The span of question text that produced this match.
    mention: str

    @property
    def param_key(self) -> str:
        """Template parameter this entity can fill: Crop -> crop_id."""
        return f"{self.label.lower()}_id"


def _candidate_phrases(question: str, max_words: int = 3) -> list[str]:
    """Generate n-grams worth looking up, longest first.

    Longest-first matters: "cotton bollworm" must be tried before "cotton", or a
    question about the pest resolves to the crop.
    """
    tokens = [t for t in _TOKEN_RE.findall(question)]
    phrases: list[str] = []

    for size in range(max_words, 0, -1):
        for start in range(len(tokens) - size + 1):
            window = tokens[start : start + size]
            if size == 1 and window[0].lower() in _STOPWORDS:
                continue
            phrase = " ".join(window)
            phrases.append(phrase)
            # Neo4j's fulltext analyzer does not stem, so "grapes" does not match
            # the stored name "Grape" and the crop silently fails to resolve.
            # Adding the singular form is cheaper and more predictable than
            # switching analyzers or fuzzy-matching.
            singular = _singularise(phrase)
            if singular != phrase:
                phrases.append(singular)

    return phrases


def _singularise(phrase: str) -> str:
    """Crude English singular of the final word. Good enough for entity names."""
    head, _, last = phrase.rpartition(" ")
    if last.endswith("ies") and len(last) > 4:
        last = last[:-3] + "y"
    elif last.endswith("ses") or last.endswith("xes") or last.endswith("ches"):
        last = last[:-2]
    elif last.endswith("s") and not last.endswith("ss") and len(last) > 3:
        last = last[:-1]
    return f"{head} {last}".strip()


def _escape_lucene(term: str) -> str:
    """Escape Lucene special characters so a question mark cannot break the query."""
    return re.sub(r'([+\-&|!(){}\[\]^"~*?:\\/])', r"\\\1", term)


async def link_entities(
    store: GraphStore,
    question: str,
    *,
    limit_per_label: int = 1,
    min_score: float = MIN_SCORE,
) -> list[EntityRef]:
    """Resolve entity mentions in ``question`` to graph nodes.

    Returns at most ``limit_per_label`` entities per label, best score first, so
    a question mentioning one crop and one region yields exactly the two ids the
    templates need.
    """
    phrases = _candidate_phrases(question)
    if not phrases:
        return []

    # One fulltext query for all candidates. Lucene OR over quoted phrases lets
    # the index do the ranking rather than issuing a round trip per n-gram.
    lucene = " OR ".join(f'"{_escape_lucene(p)}"' for p in phrases)

    rows = await store.read(
        "CALL db.index.fulltext.queryNodes($index, $query) YIELD node, score "
        "RETURN node.id AS id, node.name AS name, labels(node) AS labels, score "
        "ORDER BY score DESC LIMIT 40",
        {"index": FULLTEXT_INDEX_NAME, "query": lucene},
    )

    best_per_label: dict[str, list[EntityRef]] = {}
    lowered = question.lower()

    for row in rows:
        if row["score"] < min_score:
            continue
        label = next((lbl for lbl in row["labels"] if lbl != "Chunk"), None)
        if label is None:
            continue

        # Keep the phrase that actually appears in the question, for traceability
        # in the agent's returned state.
        name = row["name"] or ""
        mention = next(
            (p for p in phrases if p.lower() in lowered and p.lower() in name.lower()),
            name,
        )

        bucket = best_per_label.setdefault(label, [])
        if len(bucket) < limit_per_label:
            bucket.append(
                EntityRef(
                    id=row["id"],
                    name=name,
                    label=label,
                    score=round(float(row["score"]), 3),
                    mention=mention,
                )
            )

    linked = [ref for refs in best_per_label.values() for ref in refs]
    linked.sort(key=lambda r: r.score, reverse=True)

    logger.debug(
        "linked %d entities from %r: %s",
        len(linked),
        question[:60],
        [(r.label, r.id) for r in linked],
    )
    return linked


def to_params(entities: list[EntityRef]) -> dict[str, str]:
    """Collapse resolved entities into template parameters.

    ``[Crop(crop_maize), Region(reg_konya)]`` becomes
    ``{"crop_id": "crop_maize", "region_id": "reg_konya"}``.
    """
    params: dict[str, str] = {}
    for entity in entities:
        params.setdefault(entity.param_key, entity.id)
    return params
