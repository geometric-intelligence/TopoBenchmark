import copy

import torch
import torch_geometric

from topobenchmarkx.data.liftings.lifting import AbstractLifting


class KHopLifting(AbstractLifting):
    def __init__(self, k=1):
        super().__init__()
        self.k = k
        self.added_fields = ["hyperedges"]

    def forward(self, batch: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
        data_lifted = copy.copy(batch)

        n_nodes = batch.x.shape[0]

        incidence_1 = torch.zeros(n_nodes, n_nodes)
        edge_index = torch_geometric.utils.to_undirected(batch.edge_index)
        for n in range(n_nodes):
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(
                n, self.k, edge_index
            )
            incidence_1[n, neighbors] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        data_lifted[self.added_fields[0]] = incidence_1
        return data_lifted
    
    
class KNearestNeighborsLifting(AbstractLifting):
    def __init__(self, k=1):
        super().__init__()
        self.transform = torch_geometric.transforms.KNNGraph(k)
        self.added_fields = ["hyperedges"]

    def forward(self, batch: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
        batch_lifted = copy.copy(batch)
        batch_lifted.pos = batch_lifted.x
        n_nodes = batch.x.shape[0]
        incidence_1 = torch.zeros(n_nodes, n_nodes)
        data_temp = self.transform(batch_lifted)
        incidence_1[data_temp.edge_index[0],data_temp.edge_index[1]] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        batch_lifted[self.added_fields[0]] = incidence_1
        return batch_lifted