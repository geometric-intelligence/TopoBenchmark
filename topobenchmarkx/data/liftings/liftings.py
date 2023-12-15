import copy

import torch
import torch_geometric

from topobenchmarkx.data.liftings.lifting import AbstractLifting


class KHopLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, k=1):
        super().__init__()
        self.k = k
        self.added_fields = ["hyperedges"]

    def forward(self, data: torch_geometric.data.Data) -> dict:
        results = {}
        n_nodes = data.x.shape[0]
        incidence_1 = torch.zeros(n_nodes, n_nodes)
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        for n in range(n_nodes):
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(
                n, self.k, edge_index
            )
            incidence_1[n, neighbors] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        results[self.added_fields[0]] = incidence_1
        return results
    
    
class KNearestNeighborsLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, k=1):
        super().__init__()
        self.transform = torch_geometric.transforms.KNNGraph(k)
        self.added_fields = ["hyperedges"]

    def forward(self, data: torch_geometric.data.Data) -> dict:
        results = {}
        data_lifted = copy.copy(data)
        data_lifted.pos = data_lifted.x
        n_nodes = data.x.shape[0]
        incidence_1 = torch.zeros(n_nodes, n_nodes)
        data_lifted = self.transform(data_lifted)
        incidence_1[data_lifted.edge_index[0],data_lifted.edge_index[1]] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        results[self.added_fields[0]] = incidence_1
        return results