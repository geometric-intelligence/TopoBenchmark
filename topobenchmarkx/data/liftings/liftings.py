import copy

import torch
import torch_geometric

from topobenchmarkx.data.liftings.lifting import AbstractLifting


import copy
class KHopLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def forward(self, batch: torch_geometric.data.Batch, slices: dict = None) -> dict:
        batch_lifted = copy.copy(batch)

        n_nodes = batch.x.shape[0]

        incidence_1 = torch.zeros(n_nodes, n_nodes)
        edge_index = torch_geometric.utils.to_undirected(batch.edge_index)
        for n in range(n_nodes):
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(
                n, self.k, edge_index
            )
            incidence_1[n, neighbors] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        batch_lifted["hyperedges"] = incidence_1
        return batch_lifted

    def __call__(self, data, slices=None):
        return self.forward(data, slices)
    
class KNearestNeighborsLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, k=1):
        super().__init__()
        self.transform = torch_geometric.transforms.KNNGraph(k)

    def forward(self, batch: torch_geometric.data.Batch, slices: dict = None) -> dict:
        batch_lifted = copy.copy(batch)
        batch_lifted.pos = batch_lifted.x
        n_nodes = batch.x.shape[0]
        incidence_1 = torch.zeros(n_nodes, n_nodes)
        print("ok")
        print(batch_lifted.batch)
        data_temp = self.transform(batch_lifted)
        incidence_1[data_temp.edge_index[0],data_temp.edge_index[1]] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        batch_lifted["hyperedges"] = incidence_1
        return batch_lifted

    def __call__(self, data, slices=None):
        return self.forward(data, slices)
    
