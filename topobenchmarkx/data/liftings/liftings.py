import torch
import torch_geometric

from topobenchmarkx.data.liftings.lifting import AbstractLifting

import copy

class KHopLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, k=1):
        super().__init__()
        self.k = k
        
    def forward(self, data: torch_geometric.data.Data, slices: dict = None) -> dict:
        data_lifted = copy.copy(data)
        
        n_nodes = data.x.shape[0]
        
        if slices is not None:
            n_edges = data.edge_index.shape[1]
            fix_idxs = torch.zeros((2,n_edges), dtype=slices["edge_index"].dtype)
            #slices["edge_index"][-1] -= 1
            last_idx = 0
            for i, idx in enumerate(slices["edge_index"]):
                if not i==0:
                    fix_idxs[:,last_idx:idx] = slices["x"][i-1]
                    last_idx = idx
            data_lifted.edge_index += fix_idxs
            
        incidence_1 = torch.zeros(n_nodes, n_nodes)
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        for n in range(n_nodes):
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(n, self.k, edge_index)
            incidence_1[n, neighbors] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        data_lifted["hyperedges"] = incidence_1
        return data_lifted
    
    def __call__(self, data, slices = None):
        return self.forward(data, slices)