"""Graph2HypergraphLifting module."""

from .base import Graph2HypergraphLifting
from .khop import HypergraphKHopLifting
from .knn import HypergraphKNNLifting

GRAPH2HYPERGRAPH_LIFTINGS = {
    "HypergraphKHopLifting": HypergraphKHopLifting,
    "HypergraphKNNLifting": HypergraphKNNLifting,
}

__all__ = [
    "Graph2HypergraphLifting",
    "HypergraphKHopLifting",
    "HypergraphKNNLifting",
    "GRAPH2HYPERGRAPH_LIFTINGS",
]
