import copy

import torch
import torch_geometric
from scipy.optimize import minimize

__all__ = [
    "HypergraphKHopLifting",
    "HypergraphKNearestNeighborsLifting",
]


class Graph2HypergraphLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.added_fields = ["hyperedges"]
        self.type = "graph2hypergraph"

    def lift_features(
        self, data: torch_geometric.data.Data, num_hyperedges: int
    ) -> dict:
        features = {}
        features["x"] = features["x_0"] = data.x
        features["y"] = data.y
        # TODO: Projection of the features
        features["x_hyperedges"] = torch.zeros(num_hyperedges, data.x.shape[1])
        return features

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        raise NotImplementedError

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        lifted_topology = self.lift_topology(data)
        lifted_features = self.lift_features(data, lifted_topology["num_hyperedges"])
        lifted_data = torch_geometric.data.Data(**lifted_topology, **lifted_features)
        return lifted_data


class HypergraphKHopLifting(Graph2HypergraphLifting):
    def __init__(self, k_value=1, **kwargs):
        super().__init__(**kwargs)
        self.k = k_value

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        num_nodes = data.x.shape[0]

        incidence_1 = torch.zeros(num_nodes, num_nodes)
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        for n in range(num_nodes):
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(
                n, self.k, edge_index
            )
            incidence_1[n, neighbors] = 1

        num_hyperedges = incidence_1.shape[1]
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        return {"incidence_1": incidence_1, "num_hyperedges": num_hyperedges}


class HypergraphKNearestNeighborsLifting(Graph2HypergraphLifting):
    def __init__(self, k_value=1, **kwargs):
        super().__init__()
        self.k = k_value
        self.transform = torch_geometric.transforms.KNNGraph(self.k)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        num_nodes = data.x.shape[0]
        data.pos = data.x
        num_hyperedges = num_nodes
        incidence_1 = torch.zeros(num_nodes, num_nodes)
        data_lifted = self.transform(data)
        incidence_1[data_lifted.edge_index[0], data_lifted.edge_index[1]] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        return {"incidence_1": incidence_1, "num_hyperedges": num_hyperedges}
