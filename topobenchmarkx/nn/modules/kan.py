"""Original KAN, importing code from https://github.com/KindXiaoming/pykan."""

import torch
from kan import KAN as OriginalKAN


class KAN(torch.nn.Module):
    """Efficient KAN linear layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden_layers : list[int], optional
        List of hidden layer dimensions, by default None.
    grid_size : int, optional
        Number of grid points, by default 5.
    spline_order : int, optional
        Order of the spline, by default 3.
    scale_noise : float, optional
        Scale of the noise, by default 0.1.
    scale_base_mu : float, optional
        Base mu, by default 0.0.
    scale_base_sigma : float, optional
        Base sigma, by default 1.0.
    base_activation : torch.nn.SiLU, optional
        Base activation function, by default torch.nn.SiLU.
    grid_eps : float, optional
        Grid epsilon, by default 0.02.
    grid_range : tuple, optional
        Grid range, by default (-1, 1).
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
        scale_base_mu=0.0,
        scale_base_sigma=1.0,
        base_activation="silu",
        grid_eps=0.02,
        grid_range=(-1, 1),
        **kwargs,
    ):
        super().__init__()
        if hidden_layers is None:
            width = [in_channels, out_channels]
        else:
            width = [in_channels, *hidden_layers, out_channels]
        self.KAN = OriginalKAN(
            width=width,
            grid=grid_size,
            k=spline_order,
            noise_scale=scale_noise,
            scale_base_mu=scale_base_mu,
            scale_base_sigma=scale_base_sigma,
            base_fun=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.KAN(x)
