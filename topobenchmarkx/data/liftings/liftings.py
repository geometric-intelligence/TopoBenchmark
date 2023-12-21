import copy
import random
from itertools import combinations

import networkx as nx
import numpy as np
import torch
import torch_geometric
from scipy.optimize import minimize

# from topobenchmarkx.data.liftings.lifting import LiftingTransform

__all__ = [
    "IdentityLifting",
    "HypergraphKHopLifting",
    "HypergraphKNearestNeighborsLifting",
    "SimplicialNeighborhoodLifting",
    "CellCyclesLifting",
    "SimplicialCliqueLifting",
]


class IdentityLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.added_fields = []

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        return data


class Graph2HypergraphLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.added_fields = ["hyperedges"]

    def lift_features(
        self, data: torch_geometric.data.Data, num_hyperedges: int
    ) -> dict:
        features = {}
        features["x"] = features["x_0"] = data.x
        # TODO: Projection of the features
        features["x_hyperedges"] = torch.zeros(num_hyperedges, data.x.shape[1])
        return features

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        return NotImplementedError

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        lifted_topology = self.lift_topology(data)
        lifted_features = self.lift_features(data, lifted_topology["num_hyperedges"])
        lifted_data = torch_geometric.data.Data(**lifted_topology, **lifted_features)
        return lifted_data


class HypergraphKHopLifting(Graph2HypergraphLifting):
    def __init__(self, k_value=1, **kwargs):
        super().__init__(**kwargs)
        self.k = k_value

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        num_nodes = data.x.shape[0]
        num_hyperedges = num_nodes
        incidence_1 = torch.zeros(num_nodes, num_nodes)
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        for n in range(num_nodes):
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(
                n, self.k, edge_index
            )
            incidence_1[n, neighbors] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        return {"incidence_1": incidence_1, "num_hyperedges": num_hyperedges}


class HypergraphKNearestNeighborsLifting(Graph2HypergraphLifting):
    def __init__(self, k_value=1, **kwargs):
        super().__init__()
        self.k = k_value
        self.transform = torch_geometric.transforms.KNNGraph(self.k)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        num_nodes = data.x.shape[0]
        data.pos = data.x
        num_hyperedges = num_nodes
        incidence_1 = torch.zeros(num_nodes, num_nodes)
        data_lifted = self.transform(data)
        incidence_1[data_lifted.edge_index[0], data_lifted.edge_index[1]] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        return {"incidence_1": incidence_1, "num_hyperedges": num_hyperedges}


class Graph2SimplicialLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, complex_dim=2, **kwargs):
        super().__init__()
        self.complex_dim = complex_dim

    def lift_features(self, data: torch_geometric.data.Data, lifted_topology) -> dict:
        features = {}
        features["x"] = features["x_0"] = data.x
        # TODO: Projection of the features
        for i in range(self.complex_dim):
            features[f"x_{i + 1}"] = torch.zeros(
                lifted_topology[f"num_simplices_{i + 1}"], data.x.shape[1]
            )
        return features

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        return NotImplementedError

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        lifted_topology = self.lift_topology(data)
        lifted_features = self.lift_features(data, lifted_topology)
        lifted_data = torch_geometric.data.Data(**lifted_topology, **lifted_features)
        return lifted_data


class SimplicialNeighborhoodLifting(Graph2SimplicialLifting):
    def __init__(self, max_triangles=10000, **kwargs):
        super().__init__(**kwargs)
        self.max_triangles = max_triangles
        self.added_fields = []
        for i in range(1, self.complex_dim + 1):
            self.added_fields += [
                f"incidence_{i}",
                f"laplacian_down_{i}",
                f"laplacian_up_{i}",
            ]

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        lifted_topology = {}
        n_nodes = data.x.shape[0]
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        simplices = [set() for _ in range(self.complex_dim + 1)]
        for n in range(n_nodes):
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(n, 1, edge_index)
            if n not in neighbors:
                neighbors.append(n)
            neighbors = neighbors.numpy()
            neighbors = set(neighbors)
            for i in range(self.complex_dim + 1):
                for c in combinations(neighbors, i + 1):
                    simplices[i].add(tuple(c))

        for i in range(self.complex_dim + 1):
            simplices[i] = list(simplices[i])
            if i == 0:
                simplices[i].sort()
            if i == 2 and len(simplices[i]) > self.max_triangles:
                random.shuffle(simplices[i])
                simplices[i] = simplices[i][: self.max_triangles]
            lifted_topology[f"num_simplices_{i}"] = len(simplices[i])
        incidences = [
            torch.zeros(len(simplices[i]), len(simplices[i + 1]))
            for i in range(self.complex_dim)
        ]
        laplacians_up = [
            torch.zeros(len(simplices[i]), len(simplices[i]))
            for i in range(self.complex_dim)
        ]
        laplacians_down = [
            torch.zeros(len(simplices[i + 1]), len(simplices[i + 1]))
            for i in range(self.complex_dim)
        ]
        for i in range(self.complex_dim):
            for idx_i, s_i in enumerate(simplices[i]):
                for idx_i_1, s_i_1 in enumerate(simplices[i + 1]):
                    if all(e in s_i_1 for e in s_i):
                        incidences[i][idx_i][idx_i_1] = 1
            degree = torch.diag(torch.sum(incidences[i], dim=1))
            laplacians_down[i] = 2 * degree - torch.mm(
                incidences[i], torch.transpose(incidences[i], 1, 0)
            )
            degree = torch.diag(torch.sum(incidences[i], dim=0))
            laplacians_up[i] = 2 * degree - torch.mm(
                torch.transpose(incidences[i], 1, 0), incidences[i]
            )
        for i, field in enumerate(self.added_fields):
            if i % 3 == 0:
                lifted_topology[field] = incidences[int(i / 3)].to_sparse_coo()
            if i % 3 == 1:
                lifted_topology[field] = laplacians_up[int(i / 3)].to_sparse_coo()
            if i % 3 == 2:
                lifted_topology[field] = laplacians_down[int(i / 3)].to_sparse_coo()
        return lifted_topology


class SimplicialCliqueLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, complex_dim=2, **kwargs):
        super().__init__()
        self.complex_dim = complex_dim
        self.added_fields = []
        for i in range(1, complex_dim + 1):
            self.added_fields += [
                f"incidence_{i}",
                f"laplacian_down_{i}",
                f"laplacian_up_{i}",
            ]

    def forward(self, data: torch_geometric.data.Data) -> dict:
        n_nodes = data.x.shape[0]
        edges = [
            (i.item(), j.item()) for i, j in zip(data.edge_index[0], data.edge_index[1])
        ]
        G = nx.Graph()
        G.add_edges_from(edges)
        cliques = nx.find_cliques(G)
        simplices = [set() for _ in range(self.complex_dim + 1)]
        for clique in cliques:
            for i in range(self.complex_dim + 1):
                for c in combinations(clique, i + 1):
                    simplices[i].add(tuple(c))

        for i in range(self.complex_dim + 1):
            simplices[i] = list(simplices[i])
            if i == 0:
                simplices[i].sort()
        incidences = [
            torch.zeros(len(simplices[i]), len(simplices[i + 1]))
            for i in range(self.complex_dim)
        ]
        laplacians_up = [
            torch.zeros(len(simplices[i]), len(simplices[i]))
            for i in range(self.complex_dim)
        ]
        laplacians_down = [
            torch.zeros(len(simplices[i + 1]), len(simplices[i + 1]))
            for i in range(self.complex_dim)
        ]
        for i in range(self.complex_dim):
            for idx_i, s_i in enumerate(simplices[i]):
                for idx_i_1, s_i_1 in enumerate(simplices[i + 1]):
                    if all(e in s_i_1 for e in s_i):
                        incidences[i][idx_i][idx_i_1] = 1
            degree = torch.diag(torch.sum(incidences[i], dim=1))
            laplacians_down[i] = 2 * degree - torch.mm(
                incidences[i], torch.transpose(incidences[i], 1, 0)
            )
            degree = torch.diag(torch.sum(incidences[i], dim=0))
            laplacians_up[i] = 2 * degree - torch.mm(
                torch.transpose(incidences[i], 1, 0), incidences[i]
            )

        for i, field in enumerate(self.added_fields):
            if i % 3 == 0:
                data.__setitem__(field, incidences[int(i / 3)].to_sparse_coo())
            if i % 3 == 1:
                data.__setitem__(field, laplacians_up[int(i / 3)].to_sparse_coo())
            if i % 3 == 2:
                data.__setitem__(field, laplacians_down[int(i / 3)].to_sparse_coo())
        return data


class CellCyclesLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, aggregation_method="sum", **kwargs):
        super().__init__()
        self.added_fields = []
        if not aggregation_method in ["sum"]:
            raise NotImplementedError
        self.aggregation = aggregation_method
        self.added_fields = [
            "x_1",
            "incidence_1",
            "laplacian_down_1",
            "laplacian_up_1",
            "incidence_2",
            "laplacian_down_2",
            "laplacian_up_2",
        ]

    def forward(self, data: torch_geometric.data.Data) -> dict:
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

        if self.aggregation == "sum":
            x_1 = torch.mm(torch.transpose(incidence_1, 1, 0), data.x)

        data.__setitem__("incidence_1", incidence_1.to_sparse_coo())
        data.__setitem__("incidence_2", incidence_2.to_sparse_coo())
        data.__setitem__("laplacian_up_1", laplacian_up_1.to_sparse_coo())
        data.__setitem__("laplacian_up_2", laplacian_up_2.to_sparse_coo())
        data.__setitem__("laplacian_down_2", laplacian_down_2.to_sparse_coo())
        data.__setitem__("laplacian_down_1", laplacian_down_1.to_sparse_coo())
        data.__setitem__("x_1", x_1)

        return data
