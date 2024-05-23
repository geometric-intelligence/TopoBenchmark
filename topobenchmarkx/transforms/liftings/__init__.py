from topobenchmarkx.transforms.liftings.graph2cell import GRAPH2CELL_LIFTINGS  # noqa: I001
from topobenchmarkx.transforms.liftings.graph2hypergraph import GRAPH2HYPERGRAPH_LIFTINGS
from topobenchmarkx.transforms.liftings.graph2simplicial import GRAPH2SIMPLICIAL_LIFTINGS
from topobenchmarkx.transforms.liftings.lifting import (
    AbstractLifting,
    CellComplexLifting,
    CombinatorialLifting,
    GraphLifting,
    HypergraphLifting,
    PointCloudLifting,
    SimplicialLifting,
)

LIFTING_TRANSFORMS = {
    **GRAPH2CELL_LIFTINGS,
    **GRAPH2HYPERGRAPH_LIFTINGS,
    **GRAPH2SIMPLICIAL_LIFTINGS,
}

__all__ = [
    "AbstractLifting",
    "GraphLifting",
    "PointCloudLifting",
    "SimplicialLifting",
    "CellComplexLifting",
    "HypergraphLifting",
    "CombinatorialLifting",
    "LIFTING_TRANSFORMS",
]
