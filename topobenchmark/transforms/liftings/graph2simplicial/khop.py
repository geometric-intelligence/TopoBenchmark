"""This module implements the k-hop lifting of graphs to simplicial complexes."""

import random
from itertools import combinations
from typing import Any

import torch
import torch_geometric
from toponetx.classes import SimplicialComplex

from topobenchmark.transforms.liftings.base import LiftingMap


class SimplicialKHopLifting(LiftingMap):
    r"""Lift graphs to simplicial complex domain.

    The function lifts a graph to a simplicial complex by considering k-hop
    neighborhoods. For each node its neighborhood is selected and then all the
    possible simplices, when considering the neighborhood as a clique, are
    added to the simplicial complex. For this reason this lifting does not
    conserve the initial graph topology.

    Parameters
    ----------
    complex_dim : int
        Dimension of the desired complex.
    max_k_simplices : int, optional
        The maximum number of k-simplices to consider. Default is 5000.
    """

    def __init__(self, complex_dim=3, max_k_simplices=5000):
        super().__init__()
        self.complex_dim = complex_dim
        self.max_k_simplices = max_k_simplices

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_k_simplices={self.max_k_simplices!r})"

    def lift(self, domain):
        r"""Lift the topology to simplicial complex domain.

        Parameters
        ----------
        domain : nx.Graph
            Graph to be lifted.

        Returns
        -------
        toponetx.Complex
            Lifted simplicial complex.
        """
        graph = domain

        simplicial_complex = SimplicialComplex(graph)
        edge_index = torch_geometric.utils.to_undirected(
            torch.tensor(list(zip(*graph.edges, strict=False)))
        )
        simplices: list[set[tuple[Any, ...]]] = [
            set() for _ in range(2, self.complex_dim + 1)
        ]

        for n in range(graph.number_of_nodes()):
            # Find 1-hop node n neighbors
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(
                n, 1, edge_index
            )
            if n not in neighbors:
                neighbors.append(n)
            neighbors = neighbors.numpy()
            neighbors = set(neighbors)
            for i in range(1, self.complex_dim):
                for c in combinations(neighbors, i + 1):
                    simplices[i - 1].add(tuple(c))

        for set_k_simplices in simplices:
            list_k_simplices = list(set_k_simplices)
            if len(set_k_simplices) > self.max_k_simplices:
                random.shuffle(list_k_simplices)
                list_k_simplices = list_k_simplices[: self.max_k_simplices]
            simplicial_complex.add_simplices_from(list_k_simplices)

        return simplicial_complex
