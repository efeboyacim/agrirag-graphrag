"""Graph schema - the single source of truth.

Everything downstream reads this module rather than hard-coding labels:

* the CSV loader learns which file maps to which label and how to cast fields;
* the migration builder learns which constraints and indexes to create;
* Phase 2 renders :func:`schema_summary` into the router and Cypher prompts.

Adding a node label or relationship type means editing this file and nothing else.
"""

from dataclasses import dataclass, field
from typing import Literal

Cast = Literal["int", "float", "bool", "list"]

LIST_SEPARATOR = "|"


@dataclass(frozen=True)
class NodeSpec:
    """One node label and the seed file that populates it."""

    label: str
    source: str
    description: str
    casts: dict[str, Cast] = field(default_factory=dict)
    #: Properties worth exposing to an LLM writing queries against this label.
    prompt_properties: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelSpec:
    """One relationship type and the seed file that populates it."""

    type: str
    source: str
    start_label: str
    start_field: str
    end_label: str
    end_field: str
    description: str
    properties: tuple[str, ...] = ()
    casts: dict[str, Cast] = field(default_factory=dict)
    #: When set, the end label is read from this CSV column instead of
    #: :attr:`end_label`. Lets one relationship type point at several labels -
    #: RECOMMENDED_FOR targets both Crop and SoilType.
    end_label_field: str | None = None
    #: Labels the end may take when :attr:`end_label_field` is set.
    end_label_options: tuple[str, ...] = ()


NODE_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec(
        label="Crop",
        source="crops.csv",
        description="A cultivated crop species.",
        casts={"water_need_mm": "int", "aliases": "list"},
        prompt_properties=("id", "name", "scientific_name", "category", "growing_season"),
    ),
    NodeSpec(
        label="Region",
        source="regions.csv",
        description="A named agricultural production region.",
        casts={"avg_rainfall_mm": "int", "avg_temp_c": "float", "aliases": "list"},
        prompt_properties=("id", "name", "country", "avg_rainfall_mm", "avg_temp_c"),
    ),
    NodeSpec(
        label="ClimateZone",
        source="climate_zones.csv",
        description="A climate classification that a region belongs to.",
        prompt_properties=("id", "name"),
    ),
    NodeSpec(
        label="SoilType",
        source="soils.csv",
        description="A soil class with its agronomic properties.",
        casts={
            "ph_min": "float",
            "ph_max": "float",
            "organic_matter_pct": "float",
            "aliases": "list",
        },
        prompt_properties=("id", "name", "texture", "ph_min", "ph_max", "drainage"),
    ),
    NodeSpec(
        label="Pest",
        source="pests.csv",
        description=(
            "A pest, pathogen or weed. The 'category' property distinguishes insect, "
            "fungal_disease, bacterial_disease, viral_disease, weed and nematode."
        ),
        casts={"aliases": "list"},
        prompt_properties=("id", "name", "scientific_name", "category"),
    ),
    NodeSpec(
        label="Input",
        source="inputs.csv",
        description=(
            "An agricultural input: fertilizer, pesticide, fungicide, biocontrol or "
            "amendment. 'organic_approved' marks inputs permitted in certified organic "
            "production."
        ),
        casts={"organic_approved": "bool", "aliases": "list"},
        prompt_properties=(
            "id",
            "name",
            "input_type",
            "active_ingredient",
            "npk",
            "organic_approved",
        ),
    ),
    NodeSpec(
        label="Practice",
        source="practices.csv",
        description="A farming practice or technique.",
        prompt_properties=("id", "name", "category", "description"),
    ),
    NodeSpec(
        label="Regulation",
        source="regulations.csv",
        description="A regulatory or compliance instrument governing input use.",
        prompt_properties=("id", "name", "jurisdiction", "authority", "effective_date"),
    ),
)

REL_SPECS: tuple[RelSpec, ...] = (
    RelSpec(
        type="GROWN_IN",
        source="edges_grown_in.csv",
        start_label="Crop",
        start_field="crop_id",
        end_label="Region",
        end_field="region_id",
        description="The crop is commercially grown in the region.",
        properties=("suitability", "typical_yield_t_ha"),
        casts={"typical_yield_t_ha": "float"},
    ),
    RelSpec(
        type="HAS_CLIMATE",
        source="edges_has_climate.csv",
        start_label="Region",
        start_field="region_id",
        end_label="ClimateZone",
        end_field="climate_id",
        description="The region falls in this climate zone.",
    ),
    RelSpec(
        type="HAS_SOIL",
        source="edges_has_soil.csv",
        start_label="Region",
        start_field="region_id",
        end_label="SoilType",
        end_field="soil_id",
        description="The soil type occurs in the region.",
        properties=("prevalence",),
    ),
    RelSpec(
        type="SUITED_TO",
        source="edges_suited_to.csv",
        start_label="Crop",
        start_field="crop_id",
        end_label="SoilType",
        end_field="soil_id",
        description="The crop performs well on this soil type within the given pH band.",
        properties=("ph_min", "ph_max"),
        casts={"ph_min": "float", "ph_max": "float"},
    ),
    RelSpec(
        type="SUSCEPTIBLE_TO",
        source="edges_susceptible_to.csv",
        start_label="Crop",
        start_field="crop_id",
        end_label="Pest",
        end_field="pest_id",
        description="The crop is attacked by this pest, at the given severity and growth stage.",
        properties=("severity", "growth_stage"),
    ),
    RelSpec(
        type="TREATED_BY",
        source="edges_treated_by.csv",
        start_label="Pest",
        start_field="pest_id",
        end_label="Input",
        end_field="input_id",
        description=(
            "The pest is controlled by this input. 'phi_days' is the pre-harvest interval "
            "in days that must elapse between application and harvest."
        ),
        properties=("efficacy", "application_rate", "phi_days"),
        casts={"phi_days": "int"},
    ),
    RelSpec(
        type="PREVALENT_IN",
        source="edges_prevalent_in.csv",
        start_label="Pest",
        start_field="pest_id",
        end_label="Region",
        end_field="region_id",
        description="The pest is an established problem in the region during this season.",
        properties=("season",),
    ),
    RelSpec(
        type="RECOMMENDED_FOR",
        source="edges_recommended_for.csv",
        start_label="Practice",
        start_field="practice_id",
        end_label="Crop",
        end_field="target_id",
        end_label_field="target_label",
        end_label_options=("Crop", "SoilType"),
        description="The practice is recommended for this crop or soil type.",
        properties=("evidence",),
    ),
    RelSpec(
        type="MITIGATES",
        source="edges_mitigates.csv",
        start_label="Practice",
        start_field="practice_id",
        end_label="Pest",
        end_field="pest_id",
        description="The practice reduces pressure from this pest by the stated mechanism.",
        properties=("mechanism",),
    ),
    RelSpec(
        type="RESTRICTED_BY",
        source="edges_restricted_by.csv",
        start_label="Input",
        start_field="input_id",
        end_label="Regulation",
        end_field="regulation_id",
        description="Use of the input is restricted or governed by this regulation.",
    ),
    RelSpec(
        type="APPLIES_IN",
        source="edges_applies_in.csv",
        start_label="Regulation",
        start_field="regulation_id",
        end_label="Region",
        end_field="region_id",
        description="The regulation is in force in this region.",
    ),
    RelSpec(
        type="ROTATES_WITH",
        source="edges_rotates_with.csv",
        start_label="Crop",
        start_field="crop_a_id",
        end_label="Crop",
        end_field="crop_b_id",
        description=(
            "The two crops are a recommended rotation pair. Stored in one direction only - "
            "always match it undirected."
        ),
        properties=("benefit",),
    ),
)

#: Fulltext index backing entity linking (question mention -> canonical node id).
FULLTEXT_INDEX_NAME = "entity_names"
FULLTEXT_LABELS: tuple[str, ...] = (
    "Crop",
    "Region",
    "ClimateZone",
    "SoilType",
    "Pest",
    "Input",
    "Practice",
)
FULLTEXT_PROPERTIES: tuple[str, ...] = ("name", "aliases")

#: Provenance labels, populated in Phase 2 by the document extractor.
PROVENANCE_LABELS: tuple[str, ...] = ("Document", "Chunk")

NODE_SPECS_BY_LABEL: dict[str, NodeSpec] = {spec.label: spec for spec in NODE_SPECS}
REL_SPECS_BY_TYPE: dict[str, RelSpec] = {spec.type: spec for spec in REL_SPECS}

ALL_LABELS: tuple[str, ...] = tuple(spec.label for spec in NODE_SPECS)
ALL_REL_TYPES: tuple[str, ...] = tuple(spec.type for spec in REL_SPECS)


def schema_summary() -> str:
    """Render a compact schema description for LLM prompts.

    Kept terse on purpose: it is injected into the router and Cypher-generation
    prompts on every request, so tokens spent here are paid repeatedly.
    """
    lines = ["NODE LABELS"]
    for spec in NODE_SPECS:
        props = ", ".join(spec.prompt_properties)
        lines.append(f"  (:{spec.label} {{{props}}})")
        lines.append(f"      {spec.description}")

    lines.append("")
    lines.append("RELATIONSHIPS")
    for rel in REL_SPECS:
        end = "|".join(rel.end_label_options) if rel.end_label_field else rel.end_label
        props = " {" + ", ".join(rel.properties) + "}" if rel.properties else ""
        lines.append(f"  (:{rel.start_label})-[:{rel.type}{props}]->(:{end})")
        lines.append(f"      {rel.description}")
    return "\n".join(lines)
