from abc import abstractmethod

import networkx as nx
import torch_geometric
from torch_geometric.utils.undirected import is_undirected, to_undirected

from topobenchmarkx.transforms.feature_liftings.feature_liftings import (
    ConcatentionLifting,
    ProjectionSum,
    SetLifting,
)

# Implemented Feature Liftings
FEATURE_LIFTINGS = {
    "projection": ProjectionSum,
    "concatenation": ConcatentionLifting,
    "set": SetLifting,
}


class GraphLifting(torch_geometric.transforms.BaseTransform):
    r"""Abstract class for lifting graph topologies to higher-order topological
    domains.

    Args:
        feature_lifting (str, optional): The feature lifting method to be used. (default: 'projection')
        preserve_edge_attr (bool, optional): Whether to preserve edge attributes. (default: False)
        kwargs (optional): Additional arguments for the class.
    """
    def __init__(
        self, feature_lifting="projection", preserve_edge_attr=False, **kwargs
    ):
        super().__init__()
        self.feature_lifting = FEATURE_LIFTINGS[feature_lifting]()
        self.preserve_edge_attr = preserve_edge_attr
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(feature_lifting={self.feature_lifting!r}, preserve_edge_attr={self.preserve_edge_attr!r})"

    @abstractmethod
    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to higher-order topological domains.

        Args:
            data (torch_geometric.data.Data): The input data to be lifted.
        Returns:
            dict: The lifted topology.
        """
        raise NotImplementedError

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Applies the full lifting (topology + features) to the input data.

        Args:
            data (torch_geometric.data.Data): The input data.
        Returns:
            torch_geometric.data.Data: The output data.
        """
        initial_data = data.to_dict()
        lifted_topology = self.lift_topology(data)
        lifted_topology = self.feature_lifting(lifted_topology)
        lifted_data = torch_geometric.data.Data(
            **initial_data, **lifted_topology
        )
        return lifted_data

    def _data_has_edge_attr(self, data: torch_geometric.data.Data) -> bool:
        r"""Checks if the input data object has edge attributes.

        Args:
            data (torch_geometric.data.Data): The input data.
        Returns:
            bool: Whether the data object has edge attributes.
        """
        return hasattr(data, "edge_attr") and data.edge_attr is not None

    def _generate_graph_from_data(
        self, data: torch_geometric.data.Data
    ) -> nx.Graph:
        r"""Generates a NetworkX graph from the input data object.

        Args:
            data (torch_geometric.data.Data): The input data.
        Returns:
            nx.Graph: The generated NetworkX graph.
        """
        # Check if data object have edge_attr, return list of tuples as [(node_id, {'features':data}, 'dim':1)] or ??
        nodes = [
            (n, dict(features=data.x[n], dim=0))
            for n in range(data.x.shape[0])
        ]

        if self.preserve_edge_attr and self._data_has_edge_attr(data):
            # In case edge features are given, assign features to every edge
            edge_index, edge_attr = (
                data.edge_index,
                (
                    data.edge_attr
                    if is_undirected(data.edge_index, data.edge_attr)
                    else to_undirected(data.edge_index, data.edge_attr)
                ),
            )
            edges = [
                (i.item(), j.item(), dict(features=edge_attr[edge_idx], dim=1))
                for edge_idx, (i, j) in enumerate(
                    zip(edge_index[0], edge_index[1], strict=False)
                )
            ]
            self.contains_edge_attr = True
        else:
            # If edge_attr is not present, return list list of edges
            edges = [
                (i.item(), j.item())
                for i, j in zip(
                    data.edge_index[0], data.edge_index[1], strict=False
                )
            ]
            self.contains_edge_attr = False
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph
