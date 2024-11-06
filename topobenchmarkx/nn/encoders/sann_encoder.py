"""Encoder class to apply SimpleEncoder."""

import torch
import torch_geometric
from torch_geometric.nn.norm import GraphNorm

from topobenchmarkx.nn.encoders.base import AbstractFeatureEncoder


class SANNCellEncoder(AbstractFeatureEncoder):
    r"""Encoder class to apply SimpleEncoder.

    The SimpleEncoder is applied to the features of each cell
    according to a simp

    Parameters
    ----------
    in_channels : list[list[int]]
        Input dimensions for the features.
    out_channels : list[int]
        Output dimensions for the features.
    proj_dropout : float, optional
        Dropout for the BaseEncoders (default: 0).
    selected_dimensions : list[int], optional
        List of indexes to apply the BaseEncoders to (default: None).
    selected_hops : list[int], optional
        List of indexes to apply the BaseEncoders to in terms of hops (default: None).
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        proj_dropout=0,
        selected_dimensions=None,
        selected_hops=None,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimensions = (
            selected_dimensions
            if (
                selected_dimensions is not None
            )  # and len(selected_dimensions) <= len(self.in_channels))
            else range(len(self.in_channels))
        )
        self.hops = (
            selected_hops
            if (
                selected_hops is not None
            )  # and len(selected_dimensions) <= len(self.in_channels))
            else range(len(self.in_channels[0]))
        )
        for i in self.dimensions:
            for j in self.hops:
                setattr(
                    self,
                    f"encoder_{i}_{j}",
                    SimpleEncoder(
                        self.in_channels[i][j],
                        self.out_channels,
                        dropout=proj_dropout,
                    ),
                )

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, dimensions={self.dimensions})"

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Forward pass.

        The method applies the BaseEncoders to the features of the selected_dimensions.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object which should contain x_{i} features for each i in the selected_dimensions.

        Returns
        -------
        torch_geometric.data.Data
            Output data object with updated x_{i} features.
        """
        last_size = -1
        for i in self.dimensions:
            batch = getattr(data, f"batch_{i}")
            if last_size == -1:
                last_size = batch.max()
            # assert batch.max() == last_size
            for j in self.hops:
                # x_0_0 == x_0
                # x_0_1 = (dim(x_0)[0],  dim(x_0][1] * 2 + 2)
                # x_3 = x_0_1 (n, f)
                data[f"x{i}_{j}"] = getattr(self, f"encoder_{i}_{j}")(
                    data[f"x{i}_{j}"], batch
                )
        return data


class SimpleEncoder(torch.nn.Module):
    r"""SimpleEncoder used by SANN.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimensions of output features.
    dropout : float, optional
        Percentage of channels to discard between the two linear layers (default: 0).
    """

    def __init__(self, in_channels, out_channels, dropout=0):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_channels, out_channels)
        # self.linear2 = torch.nn.Linear(out_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.BN = GraphNorm(out_channels)
        # self.dropout = torch.nn.Dropout(dropout)

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.linear1.in_features}, out_channels={self.linear1.out_features})"

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the encoder.

        Applies a linear layer and a ReLu activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of dimensions [N, in_channels].
        batch : torch.Tensor
            The batch vector which assigns each element to a specific example.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [N, out_channels].
        """

        x = self.linear1(x)
        # x = self.BN(x, batch=batch) if batch.shape[0] > 0 else self.BN(x)
        return x
