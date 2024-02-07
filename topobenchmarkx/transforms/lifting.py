# from abc import ABC, abstractmethod

import torch_geometric

from topobenchmarkx.transforms.liftings.graph2cell import CellCyclesLifting
from topobenchmarkx.transforms.liftings.graph2hypergraph import (
    HypergraphKHopLifting,
    HypergraphKNearestNeighborsLifting,
)
from topobenchmarkx.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
    SimplicialNeighborhoodLifting,
)

LIFTINGS = {
    # Graph -> Hypergraph
    "HypergraphKHopLifting": HypergraphKHopLifting,
    "HypergraphKNearestNeighborsLifting": HypergraphKNearestNeighborsLifting,
    # Graph -> Simplicial Complex
    "SimplicialNeighborhoodLifting": SimplicialNeighborhoodLifting,
    "SimplicialCliqueLifting": SimplicialCliqueLifting,
    # Graph -> Cell Complex
    "CellCyclesLifting": CellCyclesLifting,
    # Identity
    "Identity": None,
}


class DataLiftingTransform(torch_geometric.transforms.BaseTransform):
    """Abstract class that provides an interface to define a custom data lifting"""

    def __init__(self, lifting, **kwargs):
        super().__init__()

        self.preserve_parameters(lifting, **kwargs)

        self.lifting = lifting
        if self.lifting == "Identity":
            self.lifting = None

        self.lifting_transform = (
            LIFTINGS[self.lifting](**kwargs) if self.lifting is not None else None
        )
        self.lifting_type = (
            self.lifting_transform.type if self.lifting is not None else None
        )

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Forward pass of the lifting"""
        lifted_data = self.lifting_transform(data) if self.lifting is not None else data
        return lifted_data

    def __call__(self, data):
        return self.forward(data)

    def preserve_parameters(self, lifting, **kwargs):
        kwargs["lifting"] = lifting
        self.parameters = kwargs


if __name__ == "__main__":
    _ = DataLiftingTransform()
