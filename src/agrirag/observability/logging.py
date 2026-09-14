"""Logging setup.

Deliberately plain in Phase 0. Phase 5 swaps the formatter for JSON and adds
the request-id / trace-id correlation fields that Portkey traces key on.
"""

import logging

_FORMAT = "%(asctime)s %(levelname)-8s %(name)-32s %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once, idempotently."""
    logging.basicConfig(level=level.upper(), format=_FORMAT, force=True)
    # The Neo4j driver is chatty at INFO about connection pooling.
    logging.getLogger("neo4j").setLevel(logging.WARNING)
