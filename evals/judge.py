"""The evaluation judge.

DeepEval's metrics need an LLM to grade with. That judge goes through the same
Portkey gateway as the application - on a *separate* config, so evaluation cost
and traffic stay distinguishable from the API's own in the traces. A nightly eval
run would otherwise make the application's request volume unreadable.

The judge config deliberately has no fallback chain. A judge that silently
degraded to a smaller model would make scores incomparable between runs, which
defeats the purpose of tracking them.

**Robustness is not optional here.** DeepEval asks for structured verdicts on
every metric, and through the gateway they come back malformed often enough to
matter. A malformed verdict makes DeepEval raise, the metric scores 0.0, and the
report reads as a quality collapse: ContextualRelevancy first measured 0.08 mean,
which looked like catastrophically irrelevant retrieval and was entirely a
serialisation problem in this file.

Two distinct causes, found in this order - the second turned out to be the
dominant one, and the first was largely a red herring built from truncated error
messages:

1. **Shape drift.** The payload arrives wrapped: as a JSON string, nested under
   the tool schema's ``$PARAMETER_NAME`` placeholder, or wrapped in its own field
   name. :func:`_coerce` unwraps these.

2. **Truncation.** The contextual metrics ask for one verdict *per statement* in
   the retrieval context, each with a written reason. At a 2000-token budget the
   model was cut off mid-tool-call and the gateway returned a tool call with
   **empty arguments** - which surfaces as "Field required", not as anything
   resembling "your response was truncated". Raising the budget took these
   metrics from 0/8 scoring to 7/8.
"""

import asyncio
import contextlib
import json
import logging
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel, ValidationError

from agrirag.config import get_settings
from agrirag.llm.portkey_client import Purpose, Span, chat_model

logger = logging.getLogger(__name__)

#: Attempts per structured call before giving up. Malformed responses are
#: intermittent, so a plain retry fixes almost all of them.
MAX_ATTEMPTS = 3

#: Output budget for a judged verdict.
#:
#: Generous on purpose. DeepEval's contextual metrics ask for one verdict *per
#: statement* in the retrieval context, each with a written reason - for ten
#: chunks that is easily several thousand tokens. Too small a budget truncates
#: the model mid-tool-call and the gateway then returns a tool call with empty
#: arguments, which surfaces as an unhelpful "Field required" validation error
#: rather than anything resembling "your response was truncated".
JUDGE_MAX_TOKENS = 8000


def _coerce(value: Any, schema: type[BaseModel]) -> BaseModel:
    """Bend a malformed response back into ``schema``, or raise.

    Handles the observed failure shapes. Anything else raises ValidationError so
    the caller can retry rather than silently scoring a metric on garbage.
    """
    if isinstance(value, schema):
        return value

    # The whole object arrived as a JSON string.
    if isinstance(value, str):
        value = json.loads(value)

    if isinstance(value, dict):
        # The real payload is sometimes nested one level down, under the tool
        # schema's own placeholder key:
        #
        #     {"$PARAMETER_NAME": {"verdicts": [...]}}
        #
        # Unwrap it. An earlier version merely *stripped* keys beginning with
        # "$", which threw the answer away and made every retry fail the same
        # way - the retry loop below could never have recovered from it, which
        # is how a systematic bug came to look like an intermittent one.
        for key in list(value):
            if key.startswith("$") and isinstance(value[key], dict):
                unwrapped = dict(value[key])
                unwrapped.update({k: v for k, v in value.items() if not k.startswith("$")})
                value = unwrapped
                break

        # Any remaining placeholder keys are echoes, not data.
        value = {k: v for k, v in value.items() if not k.startswith("$")}

        # Repair each field. The two problems compose, so they are applied in
        # order rather than as alternatives - the shape that forced this was
        #
        #     {"verdicts": '{"verdicts": [...]}'}
        #
        # a JSON *string* that parses to a dict which then still needs
        # unwrapping. Treating them as either/or fixed one shape and left
        # the other broken.
        for key in list(value):
            field_value = value[key]

            # 1. A field that should be structured arrived as a JSON string.
            if isinstance(field_value, str) and field_value.lstrip().startswith(("[", "{")):
                # Leave it alone if it does not parse - model_validate below
                # will reject it and the caller retries.
                with contextlib.suppress(json.JSONDecodeError):
                    field_value = json.loads(field_value)

            # 2. A field wrapped in its own name: {"verdicts": {"verdicts": [...]}}
            if isinstance(field_value, dict) and key in field_value:
                field_value = field_value[key]

            value[key] = field_value

    return schema.model_validate(value)


class PortkeyJudge(DeepEvalBaseLLM):
    """A DeepEval model backed by the Portkey gateway."""

    def __init__(self, trace_id: str = "eval") -> None:
        self.trace_id = trace_id
        self._settings = get_settings()
        super().__init__(model_name=self._settings.llm_model_primary)

    def load_model(self) -> "PortkeyJudge":
        return self

    def get_model_name(self) -> str:
        return f"{self._settings.llm_model_primary} (via Portkey)"

    # -- async path (what DeepEval uses in async_mode) ---------------------

    async def a_generate(
        self, prompt: str, schema: type[BaseModel] | None = None, **kwargs: Any
    ) -> Any:
        if schema is None:
            reply = await chat_model(
                span=Span.EVAL_JUDGE,
                trace_id=self.trace_id,
                purpose=Purpose.EVAL,
                max_tokens=JUDGE_MAX_TOKENS,
            ).ainvoke(prompt)
            return str(reply.content)

        # bind_tools rather than with_structured_output, so the *raw* tool-call
        # arguments are visible here.
        #
        # with_structured_output validates against the schema internally, which
        # means a malformed payload raises inside LangChain and _coerce never
        # sees the dict it knows how to repair. Reading the arguments directly is
        # what makes the repair reachable - and this is a judge, not application
        # code, so the extra handling belongs here rather than in the shared
        # factory.
        model = chat_model(
            span=Span.EVAL_JUDGE,
            trace_id=self.trace_id,
            purpose=Purpose.EVAL,
            max_tokens=JUDGE_MAX_TOKENS,
        ).bind_tools([schema], tool_choice=schema.__name__)

        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                reply = await model.ainvoke(prompt)
                calls = getattr(reply, "tool_calls", None)
                if not calls:
                    raise ValueError("judge returned no tool call")
                raw = calls[0]["args"]
                return _coerce(raw, schema)
            except (ValidationError, json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "judge returned a malformed %s (attempt %d/%d): %s | raw=%s",
                    schema.__name__,
                    attempt,
                    MAX_ATTEMPTS,
                    str(exc)[:120],
                    repr(locals().get("raw"))[:400],
                )
                await asyncio.sleep(0.5 * attempt)

        # Raise rather than return an empty verdict. A metric that errors is
        # visible in the report; one that quietly scores 0.0 is indistinguishable
        # from genuinely bad retrieval, which is the failure this whole module
        # exists to avoid.
        raise RuntimeError(
            f"judge could not produce a valid {schema.__name__} in {MAX_ATTEMPTS} attempts"
        ) from last_error

    # -- sync path ---------------------------------------------------------

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs: Any) -> Any:
        """Synchronous bridge.

        DeepEval calls this from contexts that may or may not already have a
        running loop. ``asyncio.run`` would explode inside one, so a live loop is
        handled by running the coroutine on a private loop in a worker thread.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.a_generate(prompt, schema, **kwargs))

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.a_generate(prompt, schema, **kwargs)).result()


def get_judge(trace_id: str = "eval") -> PortkeyJudge:
    """Build the judge used by every LLM-graded metric."""
    return PortkeyJudge(trace_id=trace_id)
