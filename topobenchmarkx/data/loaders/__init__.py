"""Init file for load module."""

from .graph import GRAPH_LOADERS, GRAPH_LOADERS_list
from .hypergraph import HYPERGRAPH_LOADERS, HYPERGRAPH_LOADERS_list
from .base import AbstractLoader

__all__ = [
    "AbstractLoader",
    "GRAPH_LOADERS",
    "HYPERGRAPH_LOADERS",
    *GRAPH_LOADERS_list,
    *HYPERGRAPH_LOADERS_list,
]
