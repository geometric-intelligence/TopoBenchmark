"""InfereKNNConnectivitySKLEARN class definition."""

import torch
import torch_geometric



class InferHypergraphSKLEARN(torch_geometric.transforms.BaseTransform):
    r"""Transform to infer k-nearest neighbor (k-NN) connectivity in a point cloud.

    This transform generates the k-NN connectivity of the input point cloud using
    the scikit-learn `NearestNeighbors` module. The resulting connectivity is stored
    as an edge index in the graph data.

    Parameters
    ----------
    **kwargs : optional
        Additional parameters for the transform. Supported keys:
        - metric (str): The distance metric to use (default: 'hamming').
        - n_neighbors (int): Number of neighbors to consider for k-NN (default: 5).
        - fit_key (str): The attribute key in `data` to use for fitting the k-NN model (default: 'H_tree').
        - query_key (str): The attribute key in `data` to query the k-NN model (default: 'H_tree').
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "infer_hypergraph"
        self.parameters = kwargs
        self.max_hyperedge_size = kwargs.get("max_hyperedge_size", 100)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(type={self.type!r}, "
            f"parameters={self.parameters!r})"
        )

    def forward(
        self, data: torch_geometric.data.Data
    ):
        r"""Apply the hypergraph transform to infer connectivity.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input graph data.

        Returns
        -------
        torch_geometric.data.Data
            The input data with an added edge index representing inferred connectivity.
        """

        # Add the inferred edges to the data
        # Commente out on Dec 24/24
        # data["inferred_edge_index"] = edge_index
        idx = torch.where(data['H_tree'].sum(0) < self.max_hyperedge_size)[0]
        
        # Find nodes that do not belong to any hyperedge
        node_idx = torch.where(data['H_tree'][:, idx].sum(1) == 0)[0]
        
        # Create a new hyperedge for each node 
        H_add = torch.zeros((data['H_tree'].shape[0],node_idx.shape[0]))
        # for H_add[node_idx] assign 1 for each diagonal element
        H_add[node_idx, torch.arange(node_idx.shape[0])] = 1

        H = torch.cat((data['H_tree'][:, idx], H_add), dim=1)
        data["incidence_hyperedges"] = H.to_sparse_coo()



        return data

    def get_kneighbors(self, X, n_neighbors, source_node_idx=None):
        distances, indices = self.nn_model.kneighbors(
            X=X, n_neighbors=n_neighbors
        )

        # Convert indices to edge_index format
        edge_index = self._indices_to_edge_index(indices, source_node_idx)

        # Add the inferred edges to the data
        return edge_index

    