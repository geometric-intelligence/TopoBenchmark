"""Readout function for the SANN model."""

import torch
import torch_geometric
from torch_scatter import scatter

from topobenchmarkx.nn.readouts.base import AbstractZeroCellReadOut


class SANNReadout(AbstractZeroCellReadOut):
    r"""Readout function for the SANN model.

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments. It should contain the following keys:
        - complex_dim (int): Dimension of the simplicial complex.
        - max_hop (int): Maximum hop neighbourhood to consider.
        - hidden_dim_1 (int):  Dimension of the embeddings.
        - out_channels (int): Number of classes.
        - pooling_type (str): Type of pooling operationg
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        complex_dim = kwargs["complex_dim"]
        max_hop = kwargs["max_hop"]
        hidden_dim = kwargs["hidden_dim"]
        out_channels = kwargs["out_channels"]
        pooling_type = kwargs["pooling_type"]

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(
                complex_dim * (max_hop + 1) * hidden_dim, hidden_dim
            ),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, out_channels),
        )
        assert pooling_type in ["max", "sum", "mean"], "Invalid pooling_type"
        self.pooling_type = pooling_type

    def __call__(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        """Readout logic based on model_output.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        model_out = self.forward(model_out, batch)
        model_out["logits"] = self.compute_logits(
            model_out["x_all"], batch["batch_0"]
        ).squeeze()
        return model_out

    def compute_logits(self, x: torch.Tensor, batch: torch.Tensor):
        r"""Actual computation of the logits.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        batch : torch.Tensor
            Batch tensor.

        Returns
        -------
        torch.Tensor
            Logits.
        """
        return self.linear(x)

    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        """Readout logic based on model_output.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """

        hop_tensor_names = [k for k in model_out if k.startswith("x_")]
        max_dim = len(hop_tensor_names)
        max_hop = len(model_out[hop_tensor_names[0]])

        x_all = []
        # For i-cells
        for i in range(max_dim):
            # For j-hops
            x_i_all = []
            for j in range(max_hop):
                # (dim_i_j, batch_size)
                # print(i, j)
                # print(model_out[f'x_{i}'][j].shape)
                # print(model_out[f'batch_{i}'].shape)
                x_i_j_batched = scatter(
                    model_out[f"x_{i}"][j],
                    batch[f"batch_{i}"],
                    dim=0,
                    reduce=self.pooling_type,
                )
                # print('Batch shape: ', batch[f'batch_{i}'].shape)
                # print('Batch Max node: ', batch[f'batch_{i}'].max())
                # print('Output scatter shape: ', x_i_j_batched.shape)
                x_i_all.append(x_i_j_batched)

            # (dim_i_0 + dim_i_1 + dim_i_2, batch_size)

            x_i_all_cat = torch.cat(x_i_all, dim=1)

            x_all.append(x_i_all_cat)

        # TODO: Is this fix valid ?
        lengths = set([x_i_all.shape[0] for x_i_all in x_all])
        if len(lengths) > 1:
            x_all[-1] = torch.nn.functional.pad(
                x_all[-1], (0, 0, 0, max(lengths) - x_all[-1].shape[0])
            )
        x_all_cat = torch.cat(x_all, dim=1)
        model_out["x_all"] = x_all_cat
        return model_out
