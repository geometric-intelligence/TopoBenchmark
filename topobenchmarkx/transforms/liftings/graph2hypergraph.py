from abc import abstractmethod

import torch
import torch_geometric

from topobenchmarkx.transforms.liftings.graph_lifting import GraphLifting

__all__ = [
    "HypergraphKHopLifting",
    "HypergraphKNearestNeighborsLifting",
]


class Graph2HypergraphLifting(GraphLifting):
    def __init__(self, **kwargs):
        super().__init__()
        self.type = "graph2hypergraph"

    @abstractmethod
    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        raise NotImplementedError


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
