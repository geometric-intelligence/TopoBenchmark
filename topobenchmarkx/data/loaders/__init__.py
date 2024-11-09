"""Init file for load module."""

from typing import Any
from topobenchmarkx.data.loaders.graph import GRAPH_LOADERS
from .loaders import (
    CellComplexLoader,
    GraphLoader,
    HypergraphLoader,
    SimplicialLoader,
)

LOADERS: dict[Any, Any] = {
    **GRAPH_LOADERS,
}
__all__ = [
    "GraphLoader",
    "HypergraphLoader",
    "SimplicialLoader",
    "CellComplexLoader",
    "DataLoader",
    "LOADERS",
]
