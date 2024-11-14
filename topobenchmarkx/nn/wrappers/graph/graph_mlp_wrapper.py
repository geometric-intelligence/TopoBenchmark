"""Wrapper for the GNN models."""

import torch
import torch_geometric

from topobenchmarkx.nn.wrappers.base import AbstractWrapper


class GraphMLPWrapper(AbstractWrapper):
    r"""Wrapper for the GNN models.

    This wrapper defines the forward pass of the model. The GNN models return
    the embeddings of the cells of rank 0.
    """

    def get_power_adj(self, edge_index, order=1):
        r"""Get the power of the adjacency matrix.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge index tensor.
        order : int, optional
            Order of the adjacency matrix (default: 1).

        Returns
        -------
        torch.Tensor
            Power of the adjacency matrix.
        """
        edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
        adj = torch_geometric.utils.to_dense_adj(edge_index)
        adj_power = adj.clone()
        for _ in range(order - 1):
            adj_power = torch.matmul(adj_power, adj)
        return adj_power

    def forward(self, batch):
        r"""Forward pass for the GNN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        x_0, x_dis = self.backbone(batch.x_0)

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0
        model_out["x_dis"] = x_dis
        model_out["adj_label"] = self.get_power_adj(
            batch.edge_index, self.backbone.order
        )

        return model_out
