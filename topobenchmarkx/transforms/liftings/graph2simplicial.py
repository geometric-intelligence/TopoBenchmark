import copy
import random
from itertools import combinations

import networkx as nx
import numpy as np
import torch
import torch_geometric
from scipy.optimize import minimize

__all__ = [
    "SimplicialNeighborhoodLifting",
    "SimplicialCliqueLifting",
]


class Graph2SimplicialLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, complex_dim=2, **kwargs):
        super().__init__()
        self.complex_dim = complex_dim
        self.type = "graph2simplicial"

    def lift_features(self, data: torch_geometric.data.Data, lifted_topology) -> dict:
        features = {}
        features["x"] = features["x_0"] = data.x
        features["y"] = data.y
        # TODO: Projection of the features
        for i in range(self.complex_dim):
            features[f"x_{i + 1}"] = torch.zeros(
                lifted_topology[f"num_simplices_{i + 1}"], data.x.shape[1]
            )
        return features

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        raise NotImplementedError

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


class SimplicialCliqueLifting(Graph2SimplicialLifting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
