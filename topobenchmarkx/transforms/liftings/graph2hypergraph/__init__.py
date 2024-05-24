from topobenchmarkx.transforms.liftings.graph2hypergraph.base import Graph2HypergraphLifting  # noqa: I001
from topobenchmarkx.transforms.liftings.graph2hypergraph.khop_lifting import HypergraphKHopLifting
from topobenchmarkx.transforms.liftings.graph2hypergraph.knn_lifting import HypergraphKNNLifting


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