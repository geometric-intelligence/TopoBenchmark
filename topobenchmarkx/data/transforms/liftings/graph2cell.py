import copy

import networkx as nx
import torch
import torch_geometric
from scipy.optimize import minimize

__all__ = [
    "CellCyclesLifting",
]


class Graph2CellLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, complex_dim=2, **kwargs):
        super().__init__()
        self.complex_dim = complex_dim
        self.type = "graph2cell"

    def lift_features(self, data: torch_geometric.data.Data, lifted_topology) -> dict:
        features = {}
        features["x"] = features["x_0"] = data.x
        # TODO: Projection of the features
        for i in range(self.complex_dim):
            features[f"x_{i + 1}"] = torch.zeros(
                lifted_topology[f"num_cells_{i + 1}"], data.x.shape[1]
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
        self.added_fields = [
            # "x_1",
            "incidence_1",
            "laplacian_down_1",
            "laplacian_up_1",
            "incidence_2",
            "laplacian_down_2",
            "laplacian_up_2",
        ]

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
        n_edges = len(edges)
        n_cells = len(cycles)
        incidence_1 = torch.zeros([n_nodes, n_edges])
        incidence_2 = torch.zeros([n_edges, n_cells])
        edges = [set(e) for e in edges]
        for i, edge in enumerate(edges):
            incidence_1[list(edge), i] = 1
        for i, cycle in enumerate(cycles):
            for j in range(len(cycle)):
                if j == len(cycle) - 1:
                    edge = {cycle[j], cycle[0]}
                else:
                    edge = {cycle[j], cycle[j + 1]}
                incidence_2[edges.index(edge), i] = 1
        degree = torch.diag(torch.sum(incidence_1, dim=1))
        laplacian_down_1 = 2 * degree - torch.mm(
            incidence_1, torch.transpose(incidence_1, 1, 0)
        )
        degree = torch.diag(torch.sum(incidence_2, dim=1))
        laplacian_down_2 = 2 * degree - torch.mm(
            incidence_2, torch.transpose(incidence_2, 1, 0)
        )
        degree = torch.diag(torch.sum(incidence_1, dim=0))
        laplacian_up_1 = 2 * degree - torch.mm(
            torch.transpose(incidence_1, 1, 0), incidence_1
        )
        degree = torch.diag(torch.sum(incidence_2, dim=0))
        laplacian_up_2 = 2 * degree - torch.mm(
            torch.transpose(incidence_2, 1, 0), incidence_2
        )

        # if self.aggregation == "sum":
        #    x_1 = torch.mm(torch.transpose(incidence_1, 1, 0), data.x)
        lifted_topology["num_cells_0"] = n_nodes
        lifted_topology["num_cells_1"] = n_edges
        lifted_topology["num_cells_2"] = n_cells
        lifted_topology["incidence_1"] = incidence_1.to_sparse_coo()
        lifted_topology["incidence_2"] = incidence_2.to_sparse_coo()
        lifted_topology["laplacian_up_1"] = laplacian_up_1.to_sparse_coo()
        lifted_topology["laplacian_up_2"] = laplacian_up_2.to_sparse_coo()
        lifted_topology["laplacian_down_2"] = laplacian_down_2.to_sparse_coo()
        lifted_topology["laplacian_down_1"] = laplacian_down_1.to_sparse_coo()
        # data.__setitem__("x_1", x_1)

        return lifted_topology
