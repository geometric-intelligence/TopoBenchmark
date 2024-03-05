from abc import abstractmethod

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

    def preserve_fields(self, data: torch_geometric.data.Data) -> dict:
        preserved_fields = {}
        for key, value in data.items():
            preserved_fields[key] = value
        return preserved_fields

    def lift_features(self, data: torch_geometric.data.Data, lifted_topology) -> dict:
        features = {}
        features["x_0"] = data.x
        # TODO: Projection of the features
        for i in range(self.complex_dim):
            features[f"x_{i + 1}"] = torch.zeros(
                lifted_topology["shape"][i + 1], data.x.shape[1]
            )
        return features

    @abstractmethod
    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        raise NotImplementedError

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        initial_data = self.preserve_fields(data)
        lifted_topology = self.lift_topology(data)
        lifted_features = self.lift_features(data, lifted_topology)
        lifted_data = torch_geometric.data.Data(
            **initial_data, **lifted_topology, **lifted_features
        )
        return lifted_data


class CellCyclesLifting(Graph2CellLifting):
    def __init__(self, aggregation_method="sum", **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = 2
        self.max_cell_lenght = kwargs["max_cell_lenght"]
        self.added_fields = []
        if not aggregation_method in ["sum"]:
            raise NotImplementedError
        self.aggregation = aggregation_method

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        n_nodes = data.x.shape[0]
        lifted_topology = {}
        edges = [
            (i.item(), j.item()) for i, j in zip(data.edge_index[0], data.edge_index[1])
        ]

        # Add self edges to avoid not considering isolated nodes without selfloops
        self_edges = [(i, i) for i in range(n_nodes)]
        edges = edges + self_edges

        G = nx.Graph()
        G.add_edges_from(edges)
        cycles = nx.cycle_basis(G)
        cell_complex = CellComplex(G)

        # Eliminate cycles for isolated nodes
        cycles = [cycle for cycle in cycles if len(cycle) != 1]
        # Eliminate cycles that are greater than the max_cell_lenght
        cycles = [cycle for cycle in cycles if len(cycle) <= self.max_cell_lenght]
        if len(cycles) == 0:
            lifted_topology = get_complex_connectivity(cell_complex, self.complex_dim)
            print("Warning: The graph has no cycles")
        else:
            # if len([len(cycle)==self.complex_dim for cycle in cycles]) < 3:
            #     print("Warning: The graph has less than 3 nodes, no cycles can be found")
            try:
                cell_complex.add_cells_from(cycles, rank=self.complex_dim)
            except:
                print(
                    "Warning: The graph has less than 3 nodes, no cycles can be found"
                )
            lifted_topology = get_complex_connectivity(cell_complex, self.complex_dim)
        return lifted_topology
