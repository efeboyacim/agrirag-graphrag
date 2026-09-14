"""Prompts for the agent nodes.

Kept in one module so they can be read side by side and diffed as a unit - the
router and the synthesiser have to agree on what the retrieval context looks
like, and that agreement is easier to maintain when the text sits together.
"""

from agrirag.graph.queries import TEMPLATES, template_catalogue
from agrirag.graph.schema import schema_summary

ROUTER_SYSTEM = """You route agricultural questions to the right retrieval method.

There are two retrieval paths:

GRAPH - traverses a knowledge graph of crops, regions, soils, pests, inputs,
practices and regulations. Use it when the answer depends on how entities relate
to one another: which pests attack a crop, what controls them, what is restricted
in a region, which practice suits a soil, what rotates with what. It is the only
path that can answer multi-hop questions.

SEMANTIC - similarity search over agronomy documents. Use it for definitional,
conceptual and explanatory questions: what a term means, why a mechanism works,
how a practice is carried out.

BOTH - when the question needs specific facts AND an explanation of why. Prefer
this when a question asks "why" about a relationship the graph holds.

Choose GRAPH alone when named entities and their relationships fully answer it.
Choose SEMANTIC alone when no specific crop, pest, region or input matters.

When the route includes GRAPH you must also name a template from this catalogue:

{catalogue}

The graph schema, for reference:

{schema}

Set organic_only true only when the question explicitly asks for organic,
organic-approved, or certified-organic options."""


GRADER_SYSTEM = """You judge whether retrieved context can answer a question.

Answer true only when the context contains the specific facts the question asks
for. Answer false when the context is about the right topic but does not contain
the actual answer - that distinction is the whole point of this check.

Do not use outside knowledge. Judge only what the context contains.

Always populate both fields: the boolean verdict and a one-sentence reason."""


GRADER_HUMAN = """Question: {question}

Retrieved context:
{context}

Can this context answer the question?"""


SYNTHESIS_SYSTEM = """You are an agronomy assistant answering from retrieved context.

Rules:
- Use ONLY the provided context. Do not add agronomic knowledge of your own.
- If the context does not fully answer the question, say what is missing.
- Be specific: name the crops, pests, inputs and regulations the context names,
  and carry across concrete values such as application rates and pre-harvest
  intervals when they are present.
- Mention a pre-harvest interval whenever the context gives one - it is a legal
  constraint on when the crop can be harvested, not a detail.
- Write for a farmer or agronomist: direct, practical, no preamble.
- Two to five sentences unless the question genuinely needs more.
- Do not invent citations or reference documents not present in the context."""


SYNTHESIS_HUMAN = """Question: {question}

Context:
{context}

Answer the question from this context."""


ABSTAIN_MESSAGE = (
    "I don't have enough information to answer that from the agricultural knowledge base. {detail}"
)


ROUTER_SYSTEM_COMPACT = """You route agricultural questions to the right retrieval method.

GRAPH - traverses a knowledge graph of crops, regions, soils, pests, inputs,
practices and regulations. Use it when the answer depends on how named entities
relate: which pests attack a crop, what controls them, what is restricted where,
what rotates with what.

SEMANTIC - similarity search over agronomy documents. Use it for definitional and
explanatory questions: what a term means, why a mechanism works.

BOTH - when the question needs specific facts AND an explanation of why.

When the route includes GRAPH, name one template:

{templates}

Set organic_only true only when the question explicitly asks for organic options."""


def router_system_prompt(compact: bool = False) -> str:
    """Render the router prompt with the live schema and template catalogue.

    Generated rather than hard-coded so that adding a template or a node label
    updates the prompt automatically - a catalogue that drifts from the code is
    a template the router can never choose.

    ``compact`` drops the schema dump and the per-template descriptions, taking
    the prompt from ~1670 tokens to ~250. That is not a style preference: a local
    llama3.2 crashes its runner at roughly 2000 tokens of input, and the full
    prompt sits close enough to that ceiling to take the whole request down with
    it - the router call died, the Ollama runner went with it, and the grade and
    synthesis calls that followed failed too. The symptom looked like "synthesis
    is broken" while synthesis in isolation worked perfectly.
    """
    if compact:
        names = chr(10).join(f"  - {t.name}" for t in TEMPLATES)
        return ROUTER_SYSTEM_COMPACT.format(templates=names)
    return ROUTER_SYSTEM.format(catalogue=template_catalogue(), schema=schema_summary())
