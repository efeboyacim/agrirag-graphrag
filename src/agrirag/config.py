"""Application settings.

Single source of configuration truth. Every module reads settings through
``get_settings()`` rather than touching ``os.environ`` directly, so tests can
override behaviour by clearing the cache.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed configuration. Field names map to upper-case env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["local", "ci", "prod"] = "local"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"

    # Graph store
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "agrirag_dev_pw"
    neo4j_database: str = "neo4j"

    # Vector store (embedded - a path, not a host)
    lancedb_path: str = "./data/lancedb"
    lancedb_table: str = "agri_chunks"

    # Embeddings run locally; no key, no network.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    #: Where the ONNX embedding weights are cached.
    #:
    #: Empty means the library default, which is a temporary directory. In a
    #: container that is ephemeral: the ~130 MB model re-downloads on every
    #: recreate, and a download interrupted partway leaves a cache with no
    #: config.json that fails every subsequent search with an error naming a
    #: snapshot path - nothing that points at "the model never finished
    #: downloading". Pointing this at a mounted volume fixes both.
    embedding_cache_dir: str = ""

    # LLM gateway [Phase 2]
    portkey_base_url: str = "http://localhost:8787/v1"
    portkey_api_key: str = ""
    portkey_config_app: str = ""
    portkey_config_eval: str = ""

    # Which provider the gateway routes to.
    #
    # "anthropic" is the default and what the measured evaluation baseline was
    # produced with. "ollama" routes to a local model instead - free, offline,
    # and no account - which is what makes the system demonstrable when an API
    # budget is unavailable. Quality is not comparable; see docs/portkey.md.
    llm_provider: Literal["anthropic", "ollama"] = "anthropic"

    # Local provider. The gateway runs in Docker, so it reaches an Ollama on the
    # host through host.docker.internal rather than localhost.
    ollama_host: str = "http://host.docker.internal:11434"
    ollama_model: str = "llama3.2:latest"

    #: Character budget for the retrieval context handed to a prompt.
    #:
    #: 0 means "pick a sensible default for the provider". A hosted model takes
    #: the whole context comfortably; a local 3B model does not. Measured on this
    #: machine, llama3.2 answers an ~830-token prompt in 48s and returns HTTP 500
    #: at ~2080 tokens - the runner process dies. Feeding an eleven-item context
    #: to a model with that ceiling is a design error, not a model failure.
    max_context_chars: int = 0

    #: Whether the grading node spends an LLM call.
    #:
    #: "auto" keeps it on for providers that can afford it and off for small
    #: local ones, where three LLM calls per question is the difference between
    #: an agent that completes and one that times out. Turning it off does *not*
    #: disable abstention: the empty-context branch of the grader is
    #: deterministic and still catches off-domain questions. What is lost is the
    #: narrower judgement "context exists, is on topic, but does not answer this
    #: question" - the wheat-price case in the goldens.
    llm_grading: Literal["auto", "on", "off"] = "auto"

    #: Per-request LLM timeout. 0 means "pick a sensible default for the
    #: provider" - a hosted model answers in seconds, a local one on CPU can take
    #: minutes, and a single hard-coded value cannot serve both. A 60s default
    #: made every local-model agent run fail with "Request timed out", which
    #: reads as a broken agent rather than a slow model.
    llm_timeout_seconds: int = 0

    # LLM provider. Anthropic only; the fallback chain is model-level.
    anthropic_api_key: str = ""
    llm_model_primary: str = "claude-sonnet-5"
    llm_model_fallback: str = "claude-haiku-4-5-20251001"

    # Agent threading. SQLite so a conversation survives a process restart;
    # the file lives beside the other derived data.
    checkpoint_db: str = "./data/checkpoints.sqlite"

    # Evaluation [Phase 5]. Empty => DeepEval stays fully local.
    confident_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
