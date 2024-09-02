"""Readout function for the SANN model."""

import torch
import torch_geometric

from topobenchmarkx.nn.readouts.base import AbstractZeroCellReadOut


class SANNReadout(AbstractZeroCellReadOut):
    r"""Readout function for the SANN model.

    Parameters
    ----------
    hidden_dimensions_1 : int
        Dimension of the embeddings.
    hidden_dimensions_2 : int
        Dimension of the hidden layers.
    out_channels : int
        Number of classes.
    task_level : str
        Task level.
    pooling_type : str
        Pooling type.
    """

    def __init__(
        self,
        hidden_dimensions_1,
        hidden_dimensions_2,
        out_channels,
        task_level: str,
        pooling_type: str = "sum",
    ):
        """Readout function for the zero-cell model.

        Parameters:
        -----------
        hidden_dimesions_1 : int
            Dimension of the embeddings.
        hidden_dimensions_2 : int
            Dimension of the hidden layers.
        out_channels : int
            Number of classes.
        """

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(3 * 3 * hidden_dimensions_1, hidden_dimensions_2),
            torch.functional.F.relu,
            torch.nn.Linear(hidden_dimensions_2, hidden_dimensions_2),
            torch.functional.F.relu,
            torch.nn.Linear(hidden_dimensions_2, hidden_dimensions_2),
            torch.functional.F.relu,
            torch.nn.Linear(hidden_dimensions_2, out_channels),
            torch.functional.F.sigmoid,
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
            model_out["x_0"], batch["batch_0"]
        )

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

        xi_in0 = torch.cat(
            (
                torch.sum((model_out["x0_1"]), 0),
                torch.sum((model_out["x0_2"]), 0),
                torch.sum((model_out["x0_3"]), 0),
            ),
            0,
        )
        xi_in1 = torch.cat(
            (
                torch.sum((model_out["x1_1"]), 0),
                torch.sum((model_out["x1_2"]), 0),
                torch.sum((model_out["x1_3"]), 0),
            ),
            0,
        )
        xi_in2 = torch.cat(
            (
                torch.sum((model_out["x2_1"]), 0),
                torch.sum((model_out["x2_2"]), 0),
                torch.sum((model_out["x2_3"]), 0),
            ),
            0,
        )
        # Concatenate the embeddings
        model_out["x_0"] = torch.cat(((xi_in0), (xi_in1), (xi_in2)))

        return model_out
