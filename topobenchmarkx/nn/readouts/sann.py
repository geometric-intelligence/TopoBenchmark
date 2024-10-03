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
        - hidden_dim_1 (int):  Dimension of the embeddings.
        - hidden_dim_2 (int): Dimension of the hidden layers.
        - out_channels (int): Number of classes.
        - pooling_type (str): Type of pooling operationg
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        hidden_dimensions_1 = kwargs["hidden_dim"]
        hidden_dimensions_2 = kwargs["hidden_dim_2"]
        out_channels = kwargs["out_channels"]
        pooling_type = kwargs["pooling_type"]

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(3 * 3 * hidden_dimensions_1, hidden_dimensions_2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimensions_2, hidden_dimensions_2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimensions_2, hidden_dimensions_2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimensions_2, out_channels),
            torch.nn.Softmax(dim=0),
        )  # nn.Softmax(dim=0) for multi-class

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
        )

        # Add a logit for the complement of the input
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

        x_all = []
        # Source
        for i in range(3):
            # Target
            x_i_all = []
            for j in range(3):
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
            x_i_all_cat = torch.cat(x_i_all, 1)
            # assert x_i_all_cat.shape[0] == 32, f"Expected 32, got {x_i_all_cat.shape}"
            x_all.append(x_i_all_cat)

        new_x_all = []
        try:
            x_all_cat = torch.cat(x_all, 1)
        except RuntimeError:
            last_size = -1
            for i, i_complex in enumerate(x_all):
                if i == 0:
                    last_size = i_complex.shape[0]
                    new_x_all.append(i_complex)
                    continue
                if i_complex.shape[0] < last_size:
                    new_complex = torch.cat(
                        (
                            i_complex,
                            torch.zeros(
                                last_size - i_complex.shape[0],
                                i_complex.shape[1],
                            ),
                        ),
                        dim=0,
                    )
                    new_x_all.append(new_complex)
                else:
                    new_x_all.append(i_complex)
        if len(new_x_all) > 0:
            x_all_cat = torch.cat(new_x_all, 1)
        model_out["x_all"] = x_all_cat
        return model_out
