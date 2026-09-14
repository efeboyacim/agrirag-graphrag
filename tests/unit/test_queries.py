"""Query templates and parameter binding."""

import re

import pytest

from agrirag.graph.entities import EntityRef, _candidate_phrases, to_params
from agrirag.graph.queries import (
    TEMPLATES,
    TEMPLATES_BY_NAME,
    bind_params,
    template_catalogue,
)
from agrirag.graph.schema import ALL_REL_TYPES

PARAM_RE = re.compile(r"\$(\w+)")


# --------------------------------------------------------------------------
# Template integrity
# --------------------------------------------------------------------------


def test_template_names_are_unique() -> None:
    assert len(TEMPLATES_BY_NAME) == len(TEMPLATES)


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t.name)
def test_every_cypher_parameter_is_declared(template) -> None:
    """An undeclared parameter raises at query time, not at import time - so the
    failure would surface as a mysterious 500 rather than a startup error."""
    used = set(PARAM_RE.findall(template.cypher))
    declared = set(template.required_params) | set(template.optional_params)

    assert used <= declared, f"{template.name} uses undeclared {used - declared}"


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t.name)
def test_every_declared_parameter_is_used(template) -> None:
    used = set(PARAM_RE.findall(template.cypher))
    declared = set(template.required_params) | set(template.optional_params)

    assert declared <= used, f"{template.name} declares unused {declared - used}"


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t.name)
def test_the_sentence_only_references_returned_columns(template) -> None:
    """The sentence is what reaches retrieval_context. A key the query does not
    return would raise mid-render, after the database round trip."""
    returned = set(re.findall(r"AS\s+(\w+)", template.cypher))
    referenced = set(re.findall(r"\{(\w+)\}", template.sentence))

    assert referenced <= returned, f"{template.name} renders missing {referenced - returned}"


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t.name)
def test_templates_only_traverse_declared_relationships(template) -> None:
    for rel in re.findall(r"\[:(\w+)", template.cypher):
        assert rel in ALL_REL_TYPES, f"{template.name} uses unknown relationship {rel}"


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t.name)
def test_every_template_is_described_for_the_selector(template) -> None:
    """Phase 3 selects a template from these descriptions, so an empty one is
    effectively an unreachable template."""
    assert len(template.description) > 40
    assert template.example_question


def test_the_catalogue_lists_every_template() -> None:
    catalogue = template_catalogue()
    for template in TEMPLATES:
        assert template.name in catalogue


# --------------------------------------------------------------------------
# Binding
# --------------------------------------------------------------------------


def test_missing_required_parameters_are_rejected() -> None:
    template = TEMPLATES_BY_NAME["treatments_for_crop_pests"]

    with pytest.raises(ValueError, match="crop_id"):
        bind_params(template, {})


def test_optional_parameters_default_to_none() -> None:
    """Neo4j raises on an unbound parameter, so optionals must be present as
    NULL rather than absent. That is why the templates are written as
    `$x IS NULL OR ...`."""
    template = TEMPLATES_BY_NAME["treatments_for_crop_pests"]

    bound = bind_params(template, {"crop_id": "crop_maize"})

    assert bound["crop_id"] == "crop_maize"
    assert bound["region_id"] is None
    assert bound["organic_only"] is False
    assert set(bound) >= set(PARAM_RE.findall(template.cypher))


def test_supplied_optional_parameters_win_over_defaults() -> None:
    template = TEMPLATES_BY_NAME["treatments_for_crop_pests"]

    bound = bind_params(template, {"crop_id": "crop_maize", "organic_only": True})

    assert bound["organic_only"] is True


def test_rendering_tolerates_a_null_column() -> None:
    """OPTIONAL MATCH legitimately yields NULLs; rendering must not crash."""
    template = TEMPLATES_BY_NAME["rotation_partners"]

    rendered = template.render(
        {
            "crop_name": "Wheat",
            "partner_name": "Chickpea",
            "partner_category": "legume",
            "partner_season": None,
            "benefit": "fixes nitrogen",
        }
    )

    assert "unspecified" in rendered
    assert "Chickpea" in rendered


# --------------------------------------------------------------------------
# Entity linking helpers (the parts that need no database)
# --------------------------------------------------------------------------


def test_longer_phrases_are_tried_before_shorter_ones() -> None:
    """ "cotton bollworm" must be attempted before "cotton", or a question about
    the pest resolves to the crop."""
    phrases = _candidate_phrases("pests on cotton bollworm damage")

    assert phrases.index("cotton bollworm") < phrases.index("cotton")


def test_question_words_are_not_looked_up() -> None:
    phrases = _candidate_phrases("What should I use for wheat?")

    assert "What" not in phrases
    assert "wheat" in phrases


def test_entities_collapse_into_template_parameters() -> None:
    entities = [
        EntityRef(id="crop_maize", name="Maize", label="Crop", score=4.0, mention="maize"),
        EntityRef(id="reg_konya", name="Konya", label="Region", score=3.7, mention="konya"),
    ]

    assert to_params(entities) == {"crop_id": "crop_maize", "region_id": "reg_konya"}


def test_the_highest_scoring_entity_per_label_wins() -> None:
    entities = [
        EntityRef(id="crop_maize", name="Maize", label="Crop", score=4.0, mention="maize"),
        EntityRef(id="crop_wheat", name="Wheat", label="Crop", score=1.2, mention="wheat"),
    ]

    assert to_params(entities)["crop_id"] == "crop_maize"
