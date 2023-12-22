from abc import ABC, abstractmethod

import torch_geometric

from topobenchmarkx.data.transforms.liftings.graph2cell import CellCyclesLifting
from topobenchmarkx.data.transforms.liftings.graph2hypergraph import (
    HypergraphKHopLifting,
    HypergraphKNearestNeighborsLifting,
)
from topobenchmarkx.data.transforms.liftings.graph2simplicial import (
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
}


class DataLiftingTransform(torch_geometric.transforms.BaseTransform):
    """abstract class that provides an interface to define a custom data lifting"""

    def __init__(self, lifting, **kwargs):
        super().__init__()
        self.lifting = lifting
        self.lifting_transform = (
            LIFTINGS[lifting](**kwargs) if lifting is not None else None
        )
        self.lifting_type = self.lifting_transform.type if lifting is not None else None

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Forward pass of the lifting"""
        lifted_data = self.lifting_transform(data) if self.lifting is not None else data
        return lifted_data

    def __call__(self, data):
        return self.forward(data)


if __name__ == "__main__":
    _ = DataLiftingTransform()
