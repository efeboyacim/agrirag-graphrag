"""Storage-agnostic graph interface.

The agent layer depends on this Protocol and never imports a concrete driver.
Swapping Neo4j for a managed equivalent (e.g. Neptune Analytics) means adding a
new implementation and changing one wiring line - agent logic is untouched.

Read-only by design: query-time code may only read. Ingestion uses a separate
writer path, so a compromised or hallucinated Cypher string can never mutate
the graph.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GraphStore(Protocol):
    """Minimal read surface required by the retrieval layer."""

    async def verify_connectivity(self) -> None:
        """Raise if the backing store is unreachable."""
        ...

    async def read(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a read-only query and return plain-dict rows."""
        ...

    async def close(self) -> None:
        """Release driver resources."""
        ...
