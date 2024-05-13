import torch
import torch_geometric
from torch_geometric.nn.norm import GraphNorm

from topobenchmarkx.models.abstractions.encoder import (
    AbstractInitFeaturesEncoder,
)


class BaseEncoder(torch.nn.Module):
    r"""Encoder class that uses two linear layers with GraphNorm, Relu
    activation function, and dropout between the two layers.

    Parameters
    ----------
    in_channels: int
        Dimension of input features.
    out_channels: int
        Dimensions of output features.
    dropout: float
        Percentage of channels to discard between the two linear layers.
    """

    def __init__(self, in_channels, out_channels, dropout=0):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_channels, out_channels)
        self.linear2 = torch.nn.Linear(out_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.BN = GraphNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        r"""Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of dimensions [N, in_channels].
        batch: torch.Tensor
             The batch vector which assigns each element to a specific example.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [N, out_channels].
        """
        x = self.linear1(x)
        x = self.BN(x, batch=batch) if batch.shape[0] > 0 else self.BN(x)
        x = self.dropout(self.relu(x))
        x = self.linear2(x)
        return x


class BaseFeatureEncoder(AbstractInitFeaturesEncoder):
    r"""Encoder class to apply BaseEncoder to the features of higher order
    structures.

    Parameters
    ----------
    in_channels: list(int)
        Input dimensions for the features.
    out_channels: list(int)
        Output dimensions for the features.
    proj_dropout: float
        Dropout for the BaseEncoders.
    selected_dimensions: list(int)
        List of indexes to apply the BaseEncoders to.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        proj_dropout=0,
        selected_dimensions=None,
    ):
        super(AbstractInitFeaturesEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimensions = (
            selected_dimensions
            if selected_dimensions is not None
            else range(len(self.in_channels))
        )
        for i in self.dimensions:
            setattr(
                self,
                f"encoder_{i}",
                BaseEncoder(
                    self.in_channels[i],
                    self.out_channels,
                    dropout=proj_dropout,
                ),
            )

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Forward pass.

        Parameters
        ----------
        data: torch_geometric.data.Data
            Input data object which should contain x_{i} features for each i in the selected_dimensions.

        Returns
        -------
        torch_geometric.data.Data
            Output data object.
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


# from topobenchmarkx.models.encoders.perceiver import Perceiver
# class SetFeatureEncoder(AbstractInitFeaturesEncoder):
#     r"""Encoder class to apply BaseEncoder to the node features and Perceiver to the features of higher order structures.

#     Parameters
#     ----------
#     in_channels: list(int)
#         Input dimensions for the features.
#     out_channels: list(int)
#         Output dimensions for the features.
#     proj_dropout: float
#         Dropout for the BaseEncoders.
#     selected_dimensions: list(int)
#         List of indexes to apply the BaseEncoders to.
#     """
#     def __init__(
#         self, in_channels, out_channels, proj_dropout=0, selected_dimensions=None
#     ):
#         super(AbstractInitFeaturesEncoder, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.dimensions = (
#             selected_dimensions
#             if selected_dimensions is not None
#             else range(len(self.in_channels))
#         )
#         for idx, i in enumerate(self.dimensions):
#             if idx == 0:
#                 setattr(
#                     self,
#                     f"encoder_{i}",
#                     BaseEncoder(
#                         self.in_channels[i], self.out_channels, dropout=proj_dropout
#                     ),
#                 )
#             else:
#                 setattr(
#                     self,
#                     f"encoder_{i}",
#                     Perceiver(
#                         dim=self.out_channels,
#                         depth=1,
#                         cross_heads=4,
#                         cross_dim_head=self.out_channels,
#                         latent_dim_head=self.out_channels,
#                     ),
#                 )

#     def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
#         r"""
#         Forward pass

#         Parameters
#         ----------
#         data: torch_geometric.data.Data
#             Input data object which should contain x_{i} features for each i in the selected_dimensions.

#         Returns
#         -------
#         torch_geometric.data.Data
#             Output data object.
#         """
#         if not hasattr(data, "x_0"):
#             data.x_0 = data.x

#         for idx, i in enumerate(self.dimensions):
#             if idx == 0:
#                 if hasattr(data, f"x_{i}") and hasattr(self, f"encoder_{i}"):
#                     batch = data.batch if i == 0 else getattr(data, f"batch_{i}")
#                     data[f"x_{i}"] = getattr(self, f"encoder_{i}")(
#                         data[f"x_{i}"], batch
#                     )
#             else:
#                 if hasattr(data, f"x_{i}") and hasattr(self, f"encoder_{i}"):
#                     cell_features = data["x_0"][data[f"x_{i}"].long()]
#                     data[f"x_{i}"] = getattr(self, f"encoder_{i}")(cell_features)
#                 else:
#                     data[f"x_{i}"] = torch.tensor([], device=data.x_0.device)
#         return data
