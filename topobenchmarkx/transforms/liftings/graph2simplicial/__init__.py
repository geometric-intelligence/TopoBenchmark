from topobenchmarkx.transforms.liftings.graph2simplicial.base import Graph2SimplicialLifting  # noqa: I001
from topobenchmarkx.transforms.liftings.graph2simplicial.clique_lifting import SimplicialCliqueLifting
from topobenchmarkx.transforms.liftings.graph2simplicial.khop_lifting import SimplicialKHopLifting


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
