"""Differentiable Graph Module loss function."""

import torch
import torch_geometric

from topobenchmarkx.loss.base import AbstractLoss


class DGMLoss(AbstractLoss):
    r"""DGM loss function.

    Parameters
    ----------
    loss_weight : float, optional
        Loss weight (default: 0.5).
    """

    def __init__(self, loss_weight=0.5):
        super().__init__()
        self.loss_weight = loss_weight

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> torch.Tensor:
        r"""Forward pass of the loss function.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the model output with the loss.
        """
        x_dis = model_out["x_dis"]
        # if x_dis is None:  # Validation and test
        #     return torch.tensor(0.0)

        # adj_label = self.get_power_adj(batch.edge_index)
        # graph_mlp_loss = self.loss_weight * self.graph_mlp_contrast_loss(
        #    x_dis, adj_label
        # )
        return graph_mlp_loss
