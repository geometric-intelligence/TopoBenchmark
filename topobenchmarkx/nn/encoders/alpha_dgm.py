"""Encoder class to apply BaseEncoder."""

import torch
import torch.nn as nn
from entmax import entmax15


class AlphaDGM(nn.Module):
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
        super().__init__()
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
