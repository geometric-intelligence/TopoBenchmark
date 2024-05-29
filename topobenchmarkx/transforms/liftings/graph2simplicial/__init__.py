from .base import Graph2SimplicialLifting
from .clique import SimplicialCliqueLifting
from .khop import SimplicialKHopLifting

GRAPH2SIMPLICIAL_LIFTINGS = {
    "SimplicialCliqueLifting": SimplicialCliqueLifting,
    "SimplicialKHopLifting": SimplicialKHopLifting,
}

__all__ = [
    "Graph2SimplicialLifting",
    "SimplicialCliqueLifting",
    "SimplicialKHopLifting",
    "GRAPH2SIMPLICIAL_LIFTINGS"
]
