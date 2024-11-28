"""InfereKNNConnectivitySKLEARN class definition."""

import torch
import torch_geometric
from sklearn.neighbors import NearestNeighbors


class InferKNNConnectivitySKLEARN(torch_geometric.transforms.BaseTransform):
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
        self.type = "infer_knn_connectivity_sklearn"
        self.parameters = kwargs
        self.metric = kwargs.get("metric", "hamming")
        self.n_neighbors = kwargs.get("n_neighbors", 5)
        self.fit_key = kwargs.get("fit_key", "H_tree")
        self.query_key = kwargs.get("query_key", "H_tree")
        self.nn_model = None  # Will hold the k-NN model

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(type={self.type!r}, "
            f"parameters={self.parameters!r})"
        )

    def forward(
        self, data: torch_geometric.data.Data, n_neighbors: int = None
    ):
        r"""Apply the k-NN transform to infer connectivity.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input graph data.
        n_neighbors : int, optional
            Number of neighbors to consider for k-NN. If not provided, uses `self.n_neighbors`.

        Returns
        -------
        torch_geometric.data.Data
            The input data with an added edge index representing inferred connectivity.
        """
        # Initialize and fit the k-NN model if it hasn't been done already
        if self.nn_model is None:
            self._initialize_nn_and_fit(data[self.fit_key])

        # Use the specified number of neighbors or the default
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # Query the k-NN model
        query_data = data[self.query_key]
        edge_index = self.get_kneighbors(X=query_data, n_neighbors=n_neighbors)

        # Add the inferred edges to the data
        data["inferred_edge_index"] = edge_index
        return data

    def get_kneighbors(self, X, n_neighbors, source_node_idx=None):
        distances, indices = self.nn_model.kneighbors(
            X=X, n_neighbors=n_neighbors
        )

        # Convert indices to edge_index format
        edge_index = self._indices_to_edge_index(indices, source_node_idx)

        # Add the inferred edges to the data
        return edge_index

    def _initialize_nn_and_fit(self, X):
        r"""Initialize the k-NN model and fit it to the given data.

        Parameters
        ----------
        X : array-like
            The data to fit the k-NN model on.
        """
        self.nn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors, metric=self.metric
        )
        self.nn_model.fit(X)
        print("k-NN model has been fitted successfully.")

    def _indices_to_edge_index(self, indices, source_node_idx=None):
        r"""Convert k-NN indices to a PyTorch Geometric edge index.

        Parameters
        ----------
        indices : ndarray
            Array of indices returned by the k-NN model.

        Returns
        -------
        torch.Tensor
            Edge index tensor in PyTorch Geometric format.
        """
        edge_list = []
        if source_node_idx is None:
            for source_node, neighbors in enumerate(indices):
                for target_node in neighbors:
                    if source_node != target_node:  # Avoid self-loops
                        edge_list.append([source_node, target_node])
        else:
            for source_node, neighbors in zip(source_node_idx, indices):
                for target_node in neighbors:
                    if source_node != target_node:  # Avoid self-loops
                        edge_list.append([source_node, target_node])

        # Convert edge list to a PyTorch tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Remove self-loops and make edges undirected
        edge_index = torch_geometric.utils.remove_self_loops(edge_index)[0]
        edge_index = torch_geometric.utils.to_undirected(edge_index)
        return edge_index
