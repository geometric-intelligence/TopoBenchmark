from abc import abstractmethod

import networkx as nx
import torch
import torch_geometric
from torch_geometric.utils.undirected import is_undirected, to_undirected

from topobenchmarkx.transforms.feature_liftings.feature_liftings import (
    ProjectionLifting,
)

FEATURE_LIFTINGS = {
    "projection": ProjectionLifting,
}


class GraphLifting(torch_geometric.transforms.BaseTransform):
    def __init__(
        self, feature_lifting="projection", preserve_edge_attr=False, **kwargs
    ):
        super().__init__()
        self.feature_lifting = FEATURE_LIFTINGS[feature_lifting]()
        self.preserve_edge_attr = preserve_edge_attr

    @abstractmethod
    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        raise NotImplementedError

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        initial_data = data.to_dict()
        lifted_topology = self.lift_topology(data)
        lifted_topology = self.feature_lifting(lifted_topology)
        lifted_data = torch_geometric.data.Data(**initial_data, **lifted_topology)
        return lifted_data

    def _data_has_edge_attr(self, data: torch_geometric.data.Data) -> bool:
        return hasattr(data, "edge_attr") and data.edge_attr is not None

    def _generate_graph(self, data: torch_geometric.data.Data) -> nx.Graph:
        nodes = [(n, dict(features=data.x[n], dim=0)) for n in range(data.x.shape[0])]
        if self.preserve_edge_attr and self._data_has_edge_attr(data):
            edge_index, edge_attr = data.edge_index, data.edge_attr if is_undirected(
                data.edge_index, data.edge_attr
            ) else to_undirected(data.edge_index, data.edge_attr)
            edges = [
                (i.item(), j.item(), dict(features=edge_attr[edge_idx], dim=1))
                for edge_idx, (i, j) in enumerate(zip(edge_index[0], edge_index[1]))
            ]
            self.contains_edge_attr = True
        else:
            edges = [
                (i.item(), j.item())
                for i, j in zip(data.edge_index[0], data.edge_index[1])
            ]
            self.contains_edge_attr = False
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph
