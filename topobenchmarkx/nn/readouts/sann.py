"""Readout function for the SANN model."""

import torch
import torch_geometric

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
            torch.nn.Sigmoid(),
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
            model_out["x"], batch["batch_0"]
        )

        # Add a logit for the complement of the input
        model_out["logits"] = torch.cat(
            (1 - model_out["logits"], model_out["logits"])
        ).unsqueeze(0)
        # model_out["logits"] = torch.argmax(torch.cat((1-model_out['logits'], model_out['logits']))).unsqueeze(0)
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

        # From 0-simplex to all simplex embedding
        xi_in0 = torch.cat(
            (
                torch.sum((model_out["x_0"][0]), 0),
                torch.sum((model_out["x_0"][1]), 0),
                torch.sum((model_out["x_0"][2]), 0),
            ),
            0,
        )

        # From 1-simplex to all simplex embedding
        xi_in1 = torch.cat(
            (
                torch.sum((model_out["x_1"][0]), 0),
                torch.sum((model_out["x_1"][1]), 0),
                torch.sum((model_out["x_1"][2]), 0),
            ),
            0,
        )

        # From 2-simplex to all simplex embedding
        xi_in2 = torch.cat(
            (
                torch.sum((model_out["x_2"][0]), 0),
                torch.sum((model_out["x_2"][1]), 0),
                torch.sum((model_out["x_2"][2]), 0),
            ),
            0,
        )

        # Concatenate the embeddings
        x = torch.cat(((xi_in0), (xi_in1), (xi_in2)))
        model_out["x"] = x
        return model_out
