from abc import abstractmethod

import networkx as nx
import torch
import torch_geometric
from toponetx.classes import CellComplex

from topobenchmarkx.io.load.utils import get_complex_connectivity
from topobenchmarkx.transforms.liftings.graph_lifting import GraphLifting

__all__ = [
    "CellCyclesLifting",
]


# Base
class Graph2CellLifting(GraphLifting):
    r"""Abstract class for lifting graphs to cell complexes.

    Parameters
    ----------
    complex_dim : int, optional
        The dimension of the cell complex to be generated. Default is 2.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, complex_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = complex_dim
        self.type = "graph2cell"

    @abstractmethod
    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to cell complex domain.

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

    def _get_lifted_topology(self, cell_complex: CellComplex, graph: nx.Graph) -> dict:
        r"""Returns the lifted topology.

        Parameters
        ----------
        cell_complex : CellComplex
            The cell complex.
        graph : nx.Graph
            The input graph.

        Returns
        -------
        dict
            The lifted topology.
        """
        lifted_topology = get_complex_connectivity(cell_complex, self.complex_dim)
        lifted_topology["x_0"] = torch.stack(
            list(cell_complex.get_cell_attributes("features", 0).values())
        )
        # If new edges have been added during the lifting process, we discard the edge attributes
        if self.contains_edge_attr and cell_complex.shape[1] == (
            graph.number_of_edges()
        ):
            lifted_topology["x_1"] = torch.stack(
                list(cell_complex.get_cell_attributes("features", 1).values())
            )
        return lifted_topology


class CellCyclesLifting(Graph2CellLifting):
    r"""Lifts graphs to cell complexes by identifying the cycles as 2-cells.

    Parameters
    ----------
    max_cell_length : int, optional
        The maximum length of the cycles to be lifted. Default is None.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, max_cell_length=None, **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = 2
        self.max_cell_length = max_cell_length

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Finds the cycles of a graph and lifts them to 2-cells.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        G = self._generate_graph_from_data(data)
        cycles = nx.cycle_basis(G)
        cell_complex = CellComplex(G)

        # Eliminate self-loop cycles
        cycles = [cycle for cycle in cycles if len(cycle) != 1]
        # Eliminate cycles that are greater than the max_cell_lenght
        if self.max_cell_length is not None:
            cycles = [cycle for cycle in cycles if len(cycle) <= self.max_cell_length]
        if len(cycles) != 0:
            cell_complex.add_cells_from(cycles, rank=self.complex_dim)
        return self._get_lifted_topology(cell_complex, G)
