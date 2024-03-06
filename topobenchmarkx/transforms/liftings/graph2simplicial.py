# import copy
import random
from abc import abstractmethod
from itertools import combinations

import networkx as nx
import torch
import torch_geometric
from toponetx.classes import SimplicialComplex

from topobenchmarkx.io.load.utils import get_complex_connectivity
from topobenchmarkx.transforms.liftings.graph_lifting import GraphLifting

__all__ = [
    "SimplicialNeighborhoodLifting",
    "SimplicialCliqueLifting",
]


class Graph2SimplicialLifting(GraphLifting):
    def __init__(self, complex_dim=2, **kwargs):
        super().__init__()
        self.complex_dim = complex_dim
        self.type = "graph2simplicial"

    @abstractmethod
    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        raise NotImplementedError


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
