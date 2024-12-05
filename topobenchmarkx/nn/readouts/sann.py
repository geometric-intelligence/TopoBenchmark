"""Readout function for the SANN model."""

import torch
import topomodelx
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

        self.complex_dim = kwargs["complex_dim"]
        self.max_hop = kwargs["max_hop"]
        self.task_level = kwargs["task_level"]
        hidden_dim = kwargs["hidden_dim"]
        out_channels = kwargs["out_channels"]
        pooling_type = kwargs["pooling_type"]
        self.dimensions = range(kwargs["complex_dim"] - 1, 0, -1)

        if self.task_level == "node":
            self._node_level_task_inits(hidden_dim)
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_dim, out_channels),
            )

        elif self.task_level == "graph":
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(
                    self.complex_dim * self.max_hop * hidden_dim, hidden_dim
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

        max_dim = self.complex_dim
        max_hop = self.max_hop

        if self.task_level == "graph":
            x_all = []
            # For i-cells
            for i in range(max_dim):
                # For j-hops
                x_i_all = []
                for j in range(max_hop):
                    x_i_j_batched = scatter(
                        model_out[f"x_{i}_{j}"],
                        batch[f"batch_{i}"],
                        dim=0,
                        reduce=self.pooling_type,
                    )
                    x_i_all.append(x_i_j_batched)

                x_i_all_cat = torch.cat(x_i_all, dim=1)
                x_all.append(x_i_all_cat)

            # # TODO: Is this fix valid ?
            lengths = set([x_i_all.shape[0] for x_i_all in x_all])
            if len(lengths) > 1:
                x_all[-1] = torch.nn.functional.pad(
                    x_all[-1], (0, 0, 0, max(lengths) - x_all[-1].shape[0])
                )
            x_all_cat = torch.cat(x_all, dim=1)

        elif self.task_level == "node":
            for i in self.dimensions:
                for j in range(max_hop - 1, 0, -1):
                    x_i = getattr(self, f"agg_conv_{i}")(
                        model_out[f"x_{i}_{j}"], batch[f"incidence_{i}"]
                    )
                    x_i = getattr(self, f"ln_{i}")(x_i)
                    model_out[f"x_{i-1}_{j}"] = getattr(
                        self, f"projector_{i}"
                    )(torch.cat([x_i, model_out[f"x_{i-1}_{j}"]], dim=1))

            x_all_cat = model_out[f"x_0_0"]

            # x_all = []
            # # For i-cells
            # for i in range(max_dim):
            #     # For j-hops
            #     x_i_all = []
            #     for j in range(max_hop):
            #         x_i_all.append(model_out[f"x_{i}_{j}"])

            #     x_i_all_cat = torch.cat(x_i_all, dim=1)
            #     x_all.append(x_i_all_cat)

            # x_all_cat = torch.cat(x_all, dim=1)

        model_out["x_all"] = x_all_cat  # model_out[f"x_0_0"]

        return model_out

    def _node_level_task_inits(self, hidden_dim: int):
        """Initialize the node-level task.

        Parameters
        ----------
        hidden_dim : int
            Dimension of the embeddings.
        """

        for i in self.dimensions:
            setattr(
                self,
                f"agg_conv_{i}",
                topomodelx.base.conv.Conv(
                    hidden_dim, hidden_dim, aggr_norm=False
                ),
            )

            setattr(self, f"ln_{i}", torch.nn.LayerNorm(hidden_dim))

            setattr(
                self,
                f"projector_{i}",
                torch.nn.Linear(2 * hidden_dim, hidden_dim),
            )
