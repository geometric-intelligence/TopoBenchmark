"""Encoder class to apply BaseEncoder."""

import torch
import torch.nn as nn
import torch_geometric
from entmax import entmax15

from topobenchmarkx.nn.encoders.all_cell_encoder import BaseEncoder
from topobenchmarkx.nn.encoders.base import AbstractFeatureEncoder


class DGMStructureFeatureEncoder(AbstractFeatureEncoder):
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
            base_enc = BaseEncoder(
                self.in_channels[i],
                self.out_channels,
                dropout=proj_dropout,
            )
            embed_f = BaseEncoder(
                self.in_channels[i],
                self.out_channels,
                dropout=proj_dropout,
            )

            setattr(
                self,
                f"encoder_{i}",
                DGM(base_enc=base_enc, embed_f=embed_f),
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
                x_, x_aux, edges_hat, logprobs = getattr(self, f"encoder_{i}")(
                    data[f"x_{i}"], batch
                )
                data[f"x_{i}"] = x_
                data[f"x_aux_{i}"] = x_aux
                data[f"edges_hat_{i}"] = edges_hat
                data[f"logprobs_{i}"] = logprobs
        return data


class LayerNorm(nn.Module):
    """LayerNorm with gamma and beta parameters.

    Parameters
    ----------
    gamma : torch.tensor
        Gamma parameter for the LayerNorm.
    """

    def __init__(self, gamma):
        super().__init__()
        self.gamma = nn.Parameter(gamma * torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.eps = 1e-6

    def forward(self, x):
        """LayerNorm with gamma and beta parameters.

        Parameters
        ----------
        x : torch.tensor
            Input tensor.

        Returns
        -------
        torch.tensor
            Output tensor.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        if x.size(-1) == 1:
            std = 1
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def entmax(x: torch.tensor, ln, std=0):
    """Entmax function.

    Parameters
    ----------
    x : torch.tensor
        Input tensor.

    ln : torch.tensor
        Layer normalization.

    std : float, optional
        Standard deviation for the normal distribution.

    Returns
    -------
    torch.tensor
        Output tensor.
    """
    probs = -torch.cdist(x, x)
    probs = probs + torch.empty(probs.size(), device=probs.device).normal_(
        mean=0, std=std
    )
    vprobs = entmax15(ln(probs).fill_diagonal_(-1e-6), dim=-1)
    res = (((vprobs + vprobs.t()) / 2) > 0) * 1
    edges = res.nonzero().t_()
    logprobs = res.sum(dim=1)
    return edges, logprobs


class DGM(nn.Module):
    """DGM.

    Parameters
        ----------
        base_enc : nn.Module
            Base encoder.
        embed_f : nn.Module
            Embedding function.
        gamma : float, optional
            Gamma parameter for the LayerNorm.
        std : float, optional
            Standard deviation for the normal distribution.
    """

    def __init__(
        self, base_enc: nn.Module, embed_f: nn.Module, gamma=10, std=0
    ):
        super(DGM, self).__init__()
        self.ln = LayerNorm(gamma)
        self.std = std
        self.base_enc = base_enc
        self.embed_f = embed_f

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        batch : torch.Tensor
            Batch tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        # Input embedding
        x_ = self.base_enc(x, batch)
        x_aux = self.embed_f(x, batch)
        edges_hat, logprobs = entmax(x=x_aux, ln=self.ln, std=self.std)

        return x_, x_aux, edges_hat, logprobs
