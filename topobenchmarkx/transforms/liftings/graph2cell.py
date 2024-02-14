import copy

import networkx as nx
import torch
import torch_geometric
from scipy.optimize import minimize
from toponetx.classes import CellComplex

from topobenchmarkx.io.load.utils import get_complex_connectivity

__all__ = [
    "CellCyclesLifting",
]


# Base
class Graph2CellLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, complex_dim=2, **kwargs):
        super().__init__()
        self.complex_dim = complex_dim
        self.type = "graph2cell"

    def lift_features(self, data: torch_geometric.data.Data, lifted_topology) -> dict:
        features = {}
        features["x"] = features["x_0"] = data.x
        features["y"] = data.y
        # TODO: Projection of the features
        for i in range(self.complex_dim):
            features[f"x_{i + 1}"] = torch.zeros(
                lifted_topology["shape"][i + 1], data.x.shape[1]
            )
        return features

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        raise NotImplementedError

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        lifted_topology = self.lift_topology(data)
        lifted_features = self.lift_features(data, lifted_topology)
        lifted_data = torch_geometric.data.Data(**lifted_topology, **lifted_features)
        return lifted_data


class CellCyclesLifting(Graph2CellLifting):
    def __init__(self, aggregation_method="sum", **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = 2
        self.added_fields = []
        if not aggregation_method in ["sum"]:
            raise NotImplementedError
        self.aggregation = aggregation_method

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        lifted_topology = {}
        n_nodes = data.x.shape[0]
        # edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        edges = [
            (i.item(), j.item()) for i, j in zip(data.edge_index[0], data.edge_index[1])
        ]
        G = nx.Graph()
        G.add_edges_from(edges)
        cycles = nx.cycle_basis(G)
        cell_complex = CellComplex(G)
        cell_complex.add_cells_from(cycles, rank=self.complex_dim)
        lifted_topology = get_complex_connectivity(cell_complex, self.complex_dim)
        return lifted_topology
