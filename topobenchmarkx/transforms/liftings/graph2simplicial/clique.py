"""This module implements the CliqueLifting class, which lifts graphs to simplicial complexes."""

from itertools import combinations

import networkx as nx
from toponetx.classes import SimplicialComplex

from topobenchmarkx.transforms.liftings.base import LiftingMap


class SimplicialCliqueLifting(LiftingMap):
    r"""Lift graphs to simplicial complex domain.

    The algorithm creates simplices by identifying the cliques and considering them as simplices of the same dimension.

    Parameters
    ----------
    complex_dim : int
        Maximum rank of the complex.
    """

    def __init__(self, complex_dim=2):
        super().__init__()
        # TODO: better naming
        self.complex_dim = complex_dim

    def lift(self, domain):
        r"""Lift the topology of a graph to a simplicial complex.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        graph = domain

        simplicial_complex = SimplicialComplex(graph)
        cliques = nx.find_cliques(graph)
        simplices = [set() for _ in range(2, self.complex_dim + 1)]
        for clique in cliques:
            for i in range(2, self.complex_dim + 1):
                for c in combinations(clique, i + 1):
                    simplices[i - 2].add(tuple(c))

        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        # TODO: need to check for edge preservation
        return simplicial_complex
