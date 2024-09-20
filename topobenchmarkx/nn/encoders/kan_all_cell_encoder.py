"""Class to apply BaseEncoder to the features of higher order structures."""

import torch
import torch_geometric

from topobenchmarkx.nn.encoders.base import AbstractFeatureEncoder
from topobenchmarkx.nn.kan_modules import KAN


class KANAllCellFeatureEncoder(AbstractFeatureEncoder):
    r"""Encoder class to apply BaseEncoder.

    The BaseEncoder is applied to the features of higher order
    structures. The class creates a BaseEncoder for each dimension specified in
    selected_dimensions. Then during the forward pass, the BaseEncoders are
    applied to the features of the corresponding dimensions.

    Parameters
    ----------
    in_channels : list[int]
        Input dimensions for the features.
    out_channels : list[int]
        Output dimensions for the features.
    selected_dimensions : list[int], optional
        List of indexes to apply the BaseEncoders to (default: None).
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
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
                KANBaseEncoder(
                    self.in_channels[i],
                    self.out_channels,
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
        if not hasattr(data, "x_0"):
            data.x_0 = data.x

        for i in self.dimensions:
            if hasattr(data, f"x_{i}") and hasattr(self, f"encoder_{i}"):
                batch = getattr(data, f"batch_{i}")
                data[f"x_{i}"] = getattr(self, f"encoder_{i}")(
                    data[f"x_{i}"], batch
                )
        return data


class KANBaseEncoder(torch.nn.Module):
    r"""Base encoder class used by KANAllCellFeatureEncoder.

    This class uses two KAN layers.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimensions of output features.
    hidden_layers : list[int], optional
        List of hidden layer dimensions (default: None).
    grid_size : int, optional
        Number of grid points (default: 5).
    spline_order : int, optional
        Order of the spline interpolation (default: 3).
    scale_noise : float, optional
        Scale of the noise added to the features (default: 0.1).
    scale_base : float, optional
        Scale of the base function (default: 1.0).
    scale_spline : float, optional
        Scale of the spline function (default: 1.0).
    base_activation : torch.nn.Module, optional
        Activation function for the base function (default: torch.nn.SiLU).
    grid_eps : float, optional
        Epsilon for the grid (default: 0.02).
    grid_range : tuple(float), optional
        Range of the grid (default: (-1, 1)).
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_layers=None,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=(-1, 1),
        **kwargs,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [in_channels, out_channels]
        else:
            hidden_layers = [in_channels, *hidden_layers, out_channels]
        self.KAN = KAN(
            hidden_layers,
            grid_size,
            spline_order,
            scale_noise,
            scale_base,
            scale_spline,
            base_activation,
            grid_eps,
            grid_range,
        )

    # def __repr__(self):
    #    return f"{self.__class__.__name__}(in_channels={self.linear1.in_features}, out_channels={self.linear1.out_features})"

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the encoder.

        It applies two linear layers with GraphNorm, Relu activation function, and dropout between the two layers.

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
        x = self.KAN(x)
        return x
