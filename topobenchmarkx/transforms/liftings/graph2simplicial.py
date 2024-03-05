# import copy
import random
from abc import abstractmethod
from itertools import combinations

import networkx as nx
import torch
import torch_geometric
from toponetx.classes import SimplicialComplex

from topobenchmarkx.io.load.utils import get_complex_connectivity

__all__ = [
    "SimplicialNeighborhoodLifting",
    "SimplicialCliqueLifting",
]


class Graph2SimplicialLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, complex_dim=2, **kwargs):
        super().__init__()
        self.complex_dim = complex_dim
        self.type = "graph2simplicial"

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


class SimplicialNeighborhoodLifting(Graph2SimplicialLifting):
    """ """

    def __init__(self, max_k_simplices=5000, **kwargs):
        super().__init__(**kwargs)
        self.max_k_simplices = max_k_simplices

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        n_nodes = data.x.shape[0]
        edges = [
            (i.item(), j.item()) for i, j in zip(data.edge_index[0], data.edge_index[1])
        ]
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from(edges)
        simplicial_complex = SimplicialComplex(G)
        simplices = [set() for _ in range(2, self.complex_dim + 1)]
        for n in range(n_nodes):
            # Find 1-hop node n neighbors
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(n, 1, edge_index)
            if n not in neighbors:
                neighbors.append(n)
            neighbors = neighbors.numpy()
            neighbors = set(neighbors)
            for i in range(2, self.complex_dim + 1):
                for c in combinations(neighbors, i + 1):
                    simplices[i - 2].add(tuple(c))

        for set_k_simplices in simplices:
            set_k_simplices = list(set_k_simplices)
            if len(set_k_simplices) > self.max_k_simplices:
                random.shuffle(set_k_simplices)
                set_k_simplices = set_k_simplices[: self.max_k_simplices]
            simplicial_complex.add_simplices_from(set_k_simplices)
        lifted_topology = get_complex_connectivity(simplicial_complex, self.complex_dim)
        return lifted_topology


class SimplicialCliqueLifting(Graph2SimplicialLifting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        n_nodes = data.x.shape[0]
        edges = [
            (i.item(), j.item()) for i, j in zip(data.edge_index[0], data.edge_index[1])
        ]
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from(edges)
        cliques = nx.find_cliques(G)
        simplicial_complex = SimplicialComplex(G)
        simplices = [set() for _ in range(2, self.complex_dim + 1)]
        for clique in cliques:
            for i in range(2, self.complex_dim + 1):
                for c in combinations(clique, i + 1):
                    simplices[i - 2].add(tuple(c))

        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        lifted_topology = get_complex_connectivity(simplicial_complex, self.complex_dim)
        return lifted_topology
