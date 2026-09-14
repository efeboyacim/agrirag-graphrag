"""Parameterised Cypher templates for the canonical multi-hop question shapes.

**Why templates rather than free-form text-to-Cypher.** An LLM writing arbitrary
Cypher against an 8-label schema fails in ways that are tedious to detect: it
invents a relationship direction, filters on a property that does not exist, or
returns a shape the caller cannot linearise. Selecting a template and filling
parameters turns that open-ended generation problem into a classification
problem, which is far more reliable and far cheaper to evaluate.

Free-form generation still exists as a fallback in the agent for questions no
template covers - but it is the exception, not the default path.

Each template carries a ``sentence`` format string. Rows are rendered through it
into natural language before entering ``retrieval_context``, so that graph facts
and text chunks reach the evaluation metrics in the same shape. Without this,
DeepEval's contextual metrics could not be applied uniformly across both
retrieval paths.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QueryTemplate:
    """One named, parameterised, read-only query."""

    name: str
    description: str
    cypher: str
    sentence: str
    required_params: tuple[str, ...] = ()
    optional_params: dict[str, Any] = field(default_factory=dict)
    example_question: str = ""

    def render(self, row: dict[str, Any]) -> str:
        """Turn a result row into a sentence for ``retrieval_context``."""
        safe = {k: ("unspecified" if v is None else v) for k, v in row.items()}
        try:
            return self.sentence.format(**safe)
        except KeyError as exc:  # pragma: no cover - a template authoring bug
            raise KeyError(f"template {self.name}: row is missing key {exc}") from exc


TEMPLATES: tuple[QueryTemplate, ...] = (
    QueryTemplate(
        name="treatments_for_crop_pests",
        description=(
            "Treatments available for the pests of a crop. Optionally restricted to "
            "pests actually established in a region, and/or to organic-approved inputs. "
            "Use for: what can I spray, what controls X, organic options for Y."
        ),
        required_params=("crop_id",),
        optional_params={"region_id": None, "organic_only": False},
        example_question=("Which organic-approved treatments exist for maize pests in Konya?"),
        cypher="""
        MATCH (c:Crop {id: $crop_id})-[s:SUSCEPTIBLE_TO]->(p:Pest)
        WHERE $region_id IS NULL
           OR EXISTS { MATCH (p)-[:PREVALENT_IN]->(:Region {id: $region_id}) }
        MATCH (p)-[t:TREATED_BY]->(i:Input)
        WHERE $organic_only = false OR i.organic_approved = true
        RETURN c.name AS crop_name, p.name AS pest_name, p.category AS pest_category,
               s.severity AS severity, i.name AS input_name,
               i.input_type AS input_type, i.organic_approved AS organic,
               t.efficacy AS efficacy, t.application_rate AS rate, t.phi_days AS phi_days
        ORDER BY severity DESC, pest_name, input_name
        """,
        sentence=(
            "{crop_name} is susceptible to {pest_name} ({pest_category}, {severity} severity), "
            "which is controlled by {input_name} ({input_type}, organic_approved={organic}) "
            "at {rate} with {efficacy} efficacy and a {phi_days}-day pre-harvest interval."
        ),
    ),
    QueryTemplate(
        name="restricted_inputs_for_crop_in_region",
        description=(
            "Which inputs used on a crop's pests are restricted by regulation in a given "
            "region. Use for: is X banned here, compliance, residue limits, what am I not "
            "allowed to use."
        ),
        required_params=("crop_id", "region_id"),
        example_question="Are any maize treatments restricted in Konya?",
        cypher="""
        MATCH (c:Crop {id: $crop_id})-[:SUSCEPTIBLE_TO]->(p:Pest)-[:TREATED_BY]->(i:Input)
        MATCH (i)-[:RESTRICTED_BY]->(reg:Regulation)-[:APPLIES_IN]->(r:Region {id: $region_id})
        RETURN DISTINCT c.name AS crop_name, i.name AS input_name,
               i.active_ingredient AS active_ingredient, reg.name AS regulation,
               reg.jurisdiction AS jurisdiction, reg.summary AS summary,
               r.name AS region_name
        ORDER BY input_name, regulation
        """,
        sentence=(
            "{input_name} (active ingredient {active_ingredient}), used on {crop_name} pests, "
            "is restricted in {region_name} by {regulation} ({jurisdiction}): {summary}"
        ),
    ),
    QueryTemplate(
        name="practices_for_crop_or_soil",
        description=(
            "Farming practices recommended for a crop and/or a soil type, and which pests "
            "each practice also mitigates. Use for: what should I do differently, "
            "management advice, how do I reduce pressure from X without spraying."
        ),
        required_params=("crop_id",),
        optional_params={"soil_id": None, "pest_id": None},
        example_question="What practices suit wheat on clay soils, and do any reduce rust?",
        cypher="""
        MATCH (pr:Practice)-[rec:RECOMMENDED_FOR]->(target)
        WHERE target.id = $crop_id
           OR ($soil_id IS NOT NULL AND target.id = $soil_id)
           OR ($soil_id IS NULL AND EXISTS {
                 MATCH (:Crop {id: $crop_id})-[:SUITED_TO]->(s:SoilType)
                 WHERE s.id = target.id
              })
        OPTIONAL MATCH (pr)-[m:MITIGATES]->(p:Pest)
        WHERE $pest_id IS NULL OR p.id = $pest_id
        RETURN DISTINCT pr.name AS practice_name, pr.category AS practice_category,
               pr.description AS practice_description, rec.evidence AS evidence,
               head(labels(target)) AS target_label, target.name AS target_name,
               p.name AS mitigated_pest, m.mechanism AS mechanism
        ORDER BY practice_name
        """,
        sentence=(
            "{practice_name} ({practice_category}) is recommended for {target_name} "
            "[{target_label}] with {evidence} evidence. {practice_description} "
            "It mitigates {mitigated_pest}: {mechanism}"
        ),
    ),
    QueryTemplate(
        name="treatments_for_pest",
        description=(
            "Inputs that control a named pest, independent of any crop. Optionally "
            "restricted to organic-approved inputs. Use when the question names a pest "
            "or disease but no crop: what controls X, how do I treat X."
        ),
        required_params=("pest_id",),
        optional_params={"organic_only": False},
        example_question="What controls the olive fruit fly, and which options are organic?",
        cypher="""
        MATCH (p:Pest {id: $pest_id})-[t:TREATED_BY]->(i:Input)
        WHERE $organic_only = false OR i.organic_approved = true
        RETURN p.name AS pest_name, p.category AS pest_category,
               i.name AS input_name, i.input_type AS input_type,
               i.organic_approved AS organic, t.efficacy AS efficacy,
               t.application_rate AS rate, t.phi_days AS phi_days
        ORDER BY efficacy DESC, input_name
        """,
        sentence=(
            "{pest_name} ({pest_category}) is controlled by {input_name} ({input_type}, "
            "organic_approved={organic}) at {rate} with {efficacy} efficacy and a "
            "{phi_days}-day pre-harvest interval."
        ),
    ),
    QueryTemplate(
        name="practices_for_pest",
        description=(
            "Farming practices that mitigate a named pest, and the mechanism by which "
            "they do it. Use when the question asks how to reduce pressure from a pest "
            "without naming a crop: what practices reduce X, how do I manage X "
            "non-chemically."
        ),
        required_params=("pest_id",),
        example_question="Which farming practices reduce verticillium wilt?",
        cypher="""
        MATCH (pr:Practice)-[m:MITIGATES]->(p:Pest {id: $pest_id})
        RETURN p.name AS pest_name, pr.name AS practice_name,
               pr.category AS practice_category, pr.description AS practice_description,
               m.mechanism AS mechanism
        ORDER BY practice_name
        """,
        sentence=(
            "{practice_name} ({practice_category}) mitigates {pest_name}: {mechanism} "
            "{practice_description}"
        ),
    ),
    QueryTemplate(
        name="crops_for_region_or_climate",
        description=(
            "Which crops are grown in a region, or suited to a climate zone, with yields "
            "and the soils they need. Use for: what can I grow here, is X viable in Y."
        ),
        required_params=(),
        optional_params={"region_id": None, "climate_id": None},
        example_question="Which crops grow in humid subtropical zones?",
        cypher="""
        MATCH (c:Crop)-[g:GROWN_IN]->(r:Region)
        WHERE ($region_id IS NULL OR r.id = $region_id)
          AND ($climate_id IS NULL
               OR EXISTS { MATCH (r)-[:HAS_CLIMATE]->(:ClimateZone {id: $climate_id}) })
        OPTIONAL MATCH (r)-[:HAS_CLIMATE]->(z:ClimateZone)
        RETURN c.name AS crop_name, c.category AS crop_category,
               c.growing_season AS season, c.water_need_mm AS water_need,
               r.name AS region_name, r.avg_rainfall_mm AS rainfall,
               z.name AS climate_name, g.suitability AS suitability,
               g.typical_yield_t_ha AS yield_t_ha
        ORDER BY suitability DESC, crop_name
        """,
        sentence=(
            "{crop_name} ({crop_category}, {season}, needs {water_need} mm) is grown in "
            "{region_name} ({climate_name} climate, {rainfall} mm rainfall) with "
            "{suitability} suitability and a typical yield of {yield_t_ha} t/ha."
        ),
    ),
    QueryTemplate(
        name="rotation_partners",
        description=(
            "Recommended rotation partners for a crop and why. Use for: what should follow X, "
            "break crops, rotation planning."
        ),
        required_params=("crop_id",),
        example_question="What should I rotate with wheat?",
        cypher="""
        MATCH (c:Crop {id: $crop_id})-[rot:ROTATES_WITH]-(other:Crop)
        RETURN c.name AS crop_name, other.name AS partner_name,
               other.category AS partner_category, other.growing_season AS partner_season,
               rot.benefit AS benefit
        ORDER BY partner_name
        """,
        sentence=(
            "{crop_name} rotates well with {partner_name} ({partner_category}, "
            "{partner_season}): {benefit}"
        ),
    ),
    QueryTemplate(
        name="pest_profile_for_crop",
        description=(
            "The pests affecting a crop, their severity and growth stage, where they are "
            "established, and the practices that mitigate them. Use for: what attacks X, "
            "pest pressure, what should I watch for."
        ),
        required_params=("crop_id",),
        optional_params={"region_id": None},
        example_question="What pests affect cotton in Cukurova?",
        cypher="""
        MATCH (c:Crop {id: $crop_id})-[s:SUSCEPTIBLE_TO]->(p:Pest)
        WHERE $region_id IS NULL
           OR EXISTS { MATCH (p)-[:PREVALENT_IN]->(:Region {id: $region_id}) }
        OPTIONAL MATCH (p)-[prev:PREVALENT_IN]->(r:Region)
        OPTIONAL MATCH (pr:Practice)-[m:MITIGATES]->(p)
        RETURN DISTINCT c.name AS crop_name, p.name AS pest_name,
               p.scientific_name AS scientific_name, p.category AS pest_category,
               s.severity AS severity, s.growth_stage AS growth_stage,
               r.name AS region_name, prev.season AS season,
               pr.name AS practice_name, m.mechanism AS mechanism
        ORDER BY severity DESC, pest_name
        """,
        sentence=(
            "{pest_name} ({scientific_name}, {pest_category}) attacks {crop_name} at "
            "{growth_stage} with {severity} severity; it is established in {region_name} "
            "during {season}. Mitigated by {practice_name}: {mechanism}"
        ),
    ),
    QueryTemplate(
        name="soil_requirements_for_crop",
        description=(
            "Soils a crop is suited to, the pH band it needs, and which regions have those "
            "soils. Use for: will X grow on my soil, pH questions, soil suitability."
        ),
        required_params=("crop_id",),
        example_question="What soil does hazelnut need?",
        cypher="""
        MATCH (c:Crop {id: $crop_id})-[su:SUITED_TO]->(s:SoilType)
        OPTIONAL MATCH (r:Region)-[hs:HAS_SOIL]->(s)
        RETURN DISTINCT c.name AS crop_name, s.name AS soil_name, s.texture AS texture,
               su.ph_min AS ph_min, su.ph_max AS ph_max, s.drainage AS drainage,
               s.organic_matter_pct AS organic_matter, r.name AS region_name,
               hs.prevalence AS prevalence
        ORDER BY soil_name, region_name
        """,
        sentence=(
            "{crop_name} is suited to {soil_name} ({texture}, {drainage} drainage, "
            "{organic_matter}% organic matter) within pH {ph_min}-{ph_max}; this soil is "
            "{prevalence} prevalence in {region_name}."
        ),
    ),
)

TEMPLATES_BY_NAME: dict[str, QueryTemplate] = {t.name: t for t in TEMPLATES}


def template_catalogue() -> str:
    """Render the template list for the Phase 3 selection prompt."""
    lines = []
    for template in TEMPLATES:
        params = ", ".join(template.required_params) or "none"
        optional = ", ".join(template.optional_params) or "none"
        lines.append(f"- {template.name}")
        lines.append(f"    {template.description}")
        lines.append(f"    required: {params} | optional: {optional}")
        if template.example_question:
            lines.append(f"    example: {template.example_question}")
    return "\n".join(lines)


def bind_params(template: QueryTemplate, params: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults for optional parameters and reject missing required ones.

    Neo4j raises on an unbound parameter, so every optional parameter must be
    explicitly present - as NULL when unused. That is why the templates are
    written as ``$x IS NULL OR ...`` rather than relying on absence.
    """
    missing = [p for p in template.required_params if not params.get(p)]
    if missing:
        raise ValueError(f"template {template.name} requires {missing}")

    bound = dict(template.optional_params)
    bound.update({k: v for k, v in params.items() if v is not None})
    for key in template.required_params:
        bound[key] = params[key]
    for key in template.optional_params:
        bound.setdefault(key, None)
    return bound
