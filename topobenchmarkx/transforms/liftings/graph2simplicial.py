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
    r"""Abstract class for lifting graphs to simplicial complexes.

    Parameters
    ----------
    complex_dim : int, optional
        The dimension of the simplicial complex to be generated. Default is 2.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, complex_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = complex_dim
        self.type = "graph2simplicial"
        self.signed = kwargs.get("signed", False)

    @abstractmethod
    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to simplicial complex domain.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        raise NotImplementedError

    def _get_lifted_topology(
        self, simplicial_complex: SimplicialComplex, graph: nx.Graph
    ) -> dict:
        r"""Returns the lifted topology.

        Parameters
        ----------
        simplicial_complex : SimplicialComplex
            The simplicial complex.
        graph : nx.Graph
            The input graph.

        Returns
        -------
        dict
            The lifted topology.
        """
        lifted_topology = get_complex_connectivity(
            simplicial_complex, self.complex_dim, signed=self.signed
        )
        lifted_topology["x_0"] = torch.stack(
            list(simplicial_complex.get_simplex_attributes("features", 0).values())
        )
        # If new edges have been added during the lifting process, we discard the edge attributes
        if self.contains_edge_attr and simplicial_complex.shape[1] == (
            graph.number_of_edges()
        ):
            lifted_topology["x_1"] = torch.stack(
                list(simplicial_complex.get_simplex_attributes("features", 1).values())
            )
        return lifted_topology


class SimplicialNeighborhoodLifting(Graph2SimplicialLifting):
    r"""Lifts graphs to simplicial complex domain by considering k-hop neighborhoods.

    Parameters
    ----------
    max_k_simplices : int, optional
        The maximum number of k-simplices to consider. Default is 5000.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, max_k_simplices=5000, **kwargs):
        super().__init__(**kwargs)
        self.max_k_simplices = max_k_simplices

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to simplicial complex domain by considering k-hop neighborhoods.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        graph = self._generate_graph_from_data(data)
        simplicial_complex = SimplicialComplex(graph)
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        simplices = [set() for _ in range(2, self.complex_dim + 1)]
        for n in range(graph.number_of_nodes()):
            # Find 1-hop node n neighbors
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(n, 1, edge_index)
            if n not in neighbors:
                neighbors.append(n)
            neighbors = neighbors.numpy()
            neighbors = set(neighbors)
            for i in range(1, self.complex_dim):
                for c in combinations(neighbors, i + 1):
                    simplices[i - 1].add(tuple(c))
        for set_k_simplices in simplices:
            set_k_simplices = list(set_k_simplices)
            if len(set_k_simplices) > self.max_k_simplices:
                random.shuffle(set_k_simplices)
                set_k_simplices = set_k_simplices[: self.max_k_simplices]
            simplicial_complex.add_simplices_from(set_k_simplices)
        return self._get_lifted_topology(simplicial_complex, graph)


class SimplicialCliqueLifting(Graph2SimplicialLifting):
    r"""Lifts graphs to simplicial complex domain by identifying the cliques as k-simplices.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to a simplicial complex by identifying the cliques as k-simplices.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        graph = self._generate_graph_from_data(data)
        simplicial_complex = SimplicialComplex(graph)
        cliques = nx.find_cliques(graph)
        simplices = [set() for _ in range(2, self.complex_dim + 1)]
        for clique in cliques:
            for i in range(2, self.complex_dim + 1):
                for c in combinations(clique, i + 1):
                    simplices[i - 2].add(tuple(c))

        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        return self._get_lifted_topology(simplicial_complex, graph)
