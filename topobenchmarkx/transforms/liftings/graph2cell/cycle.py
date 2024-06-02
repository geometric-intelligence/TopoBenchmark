import networkx as nx
import torch_geometric
from toponetx.classes import CellComplex

from topobenchmarkx.transforms.liftings.graph2cell.base import (
    Graph2CellLifting,
)


class CellCycleLifting(Graph2CellLifting):
    r"""Lifts graphs to cell complexes by identifying the cycles as 2-cells.

    Args:
        max_cell_length (int, optional): The maximum length of the cycles to be lifted. (default: None)
        kwargs (optional): Additional arguments for the class.
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
