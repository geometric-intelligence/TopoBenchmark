from abc import ABC, abstractmethod

import torch_geometric

from topobenchmarkx.data.liftings.liftings import *

LIFTINGS = {
    # Identity
    "IdentityLifting": IdentityLifting,
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
    """abstract class that provides an interface to define a custom lifting"""

    def __init__(self, lifting, **kwargs):
        super().__init__()
        self.lifting = LIFTINGS[lifting](**kwargs)

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Forward pass of the lifting"""
        lifted_data = self.lifting(data)
        # lifted_features = self.lift_features(data) #TODO
        return lifted_data

    def __call__(self, data):
        return self.forward(data)


if __name__ == "__main__":
    _ = LiftingTransform()
