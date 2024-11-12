"""Init file for load module."""

from typing import Any

from .graph import GRAPH_LOADERS
from .hypergaph import HYPERGRAPH_LOADERS

LOADERS: dict[Any, Any] = {
    **GRAPH_LOADERS,
    **HYPERGRAPH_LOADERS,
}
__all__ = [
    "LOADERS",
]
