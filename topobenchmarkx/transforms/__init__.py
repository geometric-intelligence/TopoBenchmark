from typing import Any

from topobenchmarkx.transforms.data_manipulations import DATA_MANIPULATIONS
from topobenchmarkx.transforms.feature_liftings import FEATURE_LIFTINGS
from topobenchmarkx.transforms.liftings.graph2cell import GRAPH2CELL_LIFTINGS
from topobenchmarkx.transforms.liftings.graph2hypergraph import (
    GRAPH2HYPERGRAPH_LIFTINGS,
)
from topobenchmarkx.transforms.liftings.graph2simplicial import (
    GRAPH2SIMPLICIAL_LIFTINGS,
)

LIFTINGS = {
    **GRAPH2CELL_LIFTINGS,
    **GRAPH2HYPERGRAPH_LIFTINGS,
    **GRAPH2SIMPLICIAL_LIFTINGS,
}

TRANSFORMS: dict[Any, Any] = {
    **LIFTINGS,
    **FEATURE_LIFTINGS,
    **DATA_MANIPULATIONS,
}

__all__ = [
    "DATA_MANIPULATIONS",
    "FEATURE_LIFTINGS",
    "LIFTINGS",
    "TRANSFORMS",
]
