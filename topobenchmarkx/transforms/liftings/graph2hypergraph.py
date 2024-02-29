from abc import abstractmethod

import torch
import torch_geometric

from topobenchmarkx.transforms.liftings.graph2domain import Graph2Domain

__all__ = [
    "HypergraphKHopLifting",
    "HypergraphKNearestNeighborsLifting",
]


# class Graph2HypergraphLifting(Graph2Domain):
#    def __init__(self, complex_dim=2, **kwargs):
#        super().__init__()
#        self.complex_dim = complex_dim
#        self.added_fields = ["hyperedges"]
#        self.type = "graph2hypergraph"


#
class Graph2HypergraphLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.type = "graph2hypergraph"

    def preserve_fields(self, data: torch_geometric.data.Data) -> dict:
        preserved_fields = {}
        for key, value in data.items():
            preserved_fields[key] = value
        return preserved_fields

    def lift_features(
        self, data: torch_geometric.data.Data, num_hyperedges: int
    ) -> dict:
        features = {}
        features["x_0"] = data.x
        # TODO: Projection of the features
        features["x_hyperedges"] = torch.zeros(num_hyperedges, data.x.shape[1])
        return features

    @abstractmethod
    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        raise NotImplementedError

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        initial_data = self.preserve_fields(data)
        lifted_topology = self.lift_topology(data)
        lifted_features = self.lift_features(data, lifted_topology["num_hyperedges"])
        lifted_data = torch_geometric.data.Data(
            **initial_data, **lifted_topology, **lifted_features
        )
        return lifted_data


class HypergraphKHopLifting(Graph2HypergraphLifting):
    def __init__(self, k_value=1, **kwargs):
        super().__init__(**kwargs)
        self.k = k_value

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        # Check if data has instance x:
        if hasattr(data, "x") and data.x is not None:
            num_nodes = data.x.shape[0]
        else:
            num_nodes = data.num_nodes

        incidence_1 = torch.zeros(num_nodes, num_nodes)
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)

        # Detect isolated nodes
        isolated_nodes = [i for i in range(num_nodes) if i not in edge_index[0]]
        if len(isolated_nodes) > 0:
            # Add completely isolated nodes to the edge_index
            edge_index = torch.cat(
                [
                    edge_index,
                    torch.tensor([isolated_nodes, isolated_nodes], dtype=torch.long),
                ],
                dim=1,
            )

        for n in range(num_nodes):
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(
                n, self.k, edge_index
            )
            incidence_1[n, neighbors] = 1

        num_hyperedges = incidence_1.shape[1]
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        return {"incidence_hyperedges": incidence_1, "num_hyperedges": num_hyperedges}


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
        return {"incidence_hyperedges": incidence_1, "num_hyperedges": num_hyperedges}
