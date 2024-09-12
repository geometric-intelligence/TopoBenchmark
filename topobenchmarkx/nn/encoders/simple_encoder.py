"""Encoder class to apply SimpleEncoder."""

import torch
import torch_geometric

from topobenchmarkx.nn.encoders.base import AbstractFeatureEncoder


class SANNCellEncoder(AbstractFeatureEncoder):
    r"""Encoder class to apply SimpleEncoder.

    The SimpleEncoder is applied to the features of each cell
    according to a simp

    Parameters
    ----------
    in_channels : list[int]
        Input dimensions for the features.
    out_channels : list[int]
        Output dimensions for the features.
    proj_dropout : float, optional
        Dropout for the BaseEncoders (default: 0).
    selected_dimensions : list[int], optional
        List of indexes to apply the BaseEncoders to (default: None).
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        proj_dropout=0,
        selected_dimensions=None,
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
        for i in self.dimensions:
            setattr(
                self,
                f"encoder_{i}",
                SimpleEncoder(
                    self.in_channels[i],
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
        self.relu = torch.nn.ReLU()

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.linear1.in_features}, out_channels={self.linear1.out_features})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the encoder.

        Applies a linear layer and a ReLu activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of dimensions [N, in_channels].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [N, out_channels].
        """
        x = self.linear1(x)
        x = self.relu(x)
        return x

        # self.in_linear_0 = torch.nn.ModuleList(
        #     torch.nn.Linear(dim_1, dim_2) for i in range(3)
        # )

        # # k-simplex to 1-simplex (k=0,1,2)
        # self.in_linear_1 = torch.nn.ModuleList(
        #     [
        #         torch.nn.Linear(6, dim_2),
        #         torch.nn.Linear(12, dim_2),
        #         torch.nn.Linear(9, dim_2),
        #     ]
        # )

        # # k-simplex to 2-simplex (k=0,1,2)
        # self.in_linear_2 = torch.nn.ModuleList(
        #     [
        #         torch.nn.Linear(18, dim_2),
        #         torch.nn.Linear(39, dim_2),
        #         torch.nn.Linear(30, dim_2),
        #     ]
        # )
