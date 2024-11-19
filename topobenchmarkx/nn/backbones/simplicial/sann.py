"""SANN network."""

import torch
import torch.nn.functional
from torch.nn.parameter import Parameter


class SANN(torch.nn.Module):
    r"""SANN network.

    Parameters
    ----------
    in_channels : tuple of int or int
        Dimension of the hidden layers.
    hidden_channels : int
        Dimension of the output layer.
    update_func : str
        Update function.
    complex_dim : int
        Dimension of the complex.
    hop_num : int
        Number of hops.
    n_layers : int
        Number of layers.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        update_func=None,
        complex_dim=3,
        hop_num=2,
        n_layers=2,
    ):
        super().__init__()
        self.complex_dim = complex_dim
        self.hop_num = hop_num

        assert n_layers >= 1

        if isinstance(in_channels, int):  # If only one value is passed
            in_channels = [in_channels] * self.complex_dim

        self.layers = torch.nn.ModuleList()

        # Set of simplices layers
        self.layers_0 = torch.nn.ModuleList(
            SANNLayer(
                [in_channels[i] for i in range(complex_dim)],
                [hidden_channels] * complex_dim,
                update_func=update_func,
            )
            for i in range(complex_dim)
        )
        self.layers.append(self.layers_0)

        # From layer 1 to n_layers
        for i in range(1, n_layers):
            self.layers.append(
                torch.nn.ModuleList(
                    SANNLayer(
                        [hidden_channels] * complex_dim,
                        [hidden_channels] * complex_dim,
                        update_func="lrelu",
                    )
                    for i in range(complex_dim)
                )
            )

    def forward(self, x):
        r"""Forward pass of the model.

        Parameters
        ----------
        x : tuple(tuple(torch.Tensor))
            Tuple of tuple containing the input tensors for each simplex.

        Returns
        -------
        tuple(tuple(torch.Tensor))
            Tuple of tuples of final hidden state tensors.
        """

        # The follwing line will mean the same as:
        # # For each k: 0 to k (k=0,1,2)
        # x_0_tup = tuple(self.in_linear_0[i](x[0][i]) for i in range(3))
        # # For each k: 1 to k (k=0,1,2)
        # x_1_tup = tuple(self.in_linear_1[i](x[1][i]) for i in range(3))
        # # For each k: 2 to k (k=0,1,2)
        # x_2_tup = tuple(self.in_linear_2[i](x[2][i]) for i in range(3))

        # For each layer in the network
        for layer in self.layers:
            # Temporary list
            x_i = list()

            # For each i-simplex (i=0,1,2) to all other k-simplices
            for i in range(3):
                # Goes from i-simplex to all other simplices k<=i
                x_i_to_t = layer[i](x[i])
                # Update the i-th simplex to all other simplices embeddings
                x_i.append(tuple(x_i_to_t))
            x = tuple(x_i)
        return x


class SANNLayer(torch.nn.Module):
    r"""One layer in the SANN architecture.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    aggr_norm : bool
        Whether to perform aggregation normalization.
    update_func : str
        Update function.
    initialization : str
        Initialization method.

    Returns
    -------
    torch.Tensor
        Output
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        aggr_norm: bool = False,
        update_func=None,
        initialization: str = "xavier_normal",
    ) -> None:
        super().__init__()

        in_channels_0, in_channels_1, in_channels_2 = in_channels
        out_channels_0, out_channels_1, out_channels_2 = out_channels

        self.in_channels_0 = in_channels_0
        self.in_channels_1 = in_channels_1
        self.in_channels_2 = in_channels_2
        self.out_channels_0 = out_channels_0
        self.out_channels_1 = out_channels_1
        self.out_channels_2 = out_channels_2

        self.aggr_norm = aggr_norm
        self.update_func = update_func
        self.initialization = initialization

        assert initialization in ["xavier_uniform", "xavier_normal"]

        self.weight_0 = Parameter(
            torch.Tensor(
                self.in_channels_0,
                self.out_channels_0,
            )
        )

        self.weight_1 = Parameter(
            torch.Tensor(
                self.in_channels_1,
                self.out_channels_1,
            )
        )
        self.weight_2 = Parameter(
            torch.Tensor(
                self.in_channels_2,
                self.out_channels_2,
            )
        )

        self.biases_0 = Parameter(torch.Tensor(self.out_channels_0))
        self.biases_1 = Parameter(torch.Tensor(self.out_channels_1))
        self.biases_2 = Parameter(torch.Tensor(self.out_channels_2))

        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.414):
        r"""Reset learnable parameters.

        Parameters
        ----------
        gain : float
            Gain for the weight initialization.
        """
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.weight_0, gain=gain)
            torch.nn.init.xavier_uniform_(self.weight_1, gain=gain)
            torch.nn.init.xavier_uniform_(self.weight_2, gain=gain)

            torch.nn.init.zeros_(self.biases_0)
            torch.nn.init.zeros_(self.biases_1)
            torch.nn.init.zeros_(self.biases_2)
        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.weight_0, gain=gain)
            torch.nn.init.xavier_normal_(self.weight_1, gain=gain)
            torch.nn.init.xavier_normal_(self.weight_2, gain=gain)

            torch.nn.init.zeros_(self.biases_0)
            torch.nn.init.zeros_(self.biases_1)
            torch.nn.init.zeros_(self.biases_2)
        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def update(self, x: torch.Tensor):
        """Update embeddings on each cell (step 4).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Updated tensor.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x)
        if self.update_func == "lrelu":
            return torch.nn.functional.leaky_relu(x)
        return None

    def forward(self, x_all: dict[int, torch.Tensor]):
        r"""Forward computation.

        Parameters
        ----------
        x_all : Dict[Int, torch.Tensor]
            Dictionary of tensors where each simplex dimension (node, edge, face) represents a key, 0-indexed.

        Returns
        -------
        torch.Tensor
            Output tensors for each 0-cell.
        torch.Tensor
            Output tensors for each 1-cell.
        torch.Tensor
            Output tensors for each 2-cell.
        """
        # Extract all cells to all cells
        x_0 = x_all[0]
        x_1 = x_all[1]
        x_2 = x_all[2]

        # k-simplex to k-simplex
        x_k_to_0 = torch.mm(x_0, self.weight_0)
        x_k_to_1 = torch.mm(x_1, self.weight_1)
        x_k_to_2 = torch.mm(x_2, self.weight_2)

        # TODO Check aggregation as list of ys
        # Need to check that this einsums are correct
        # y_0 = torch.einsum("nik,iok->no", x_0_all, self.weight_0)
        # y_1 = torch.einsum("nik,iok->no", x_1_all, self.weight_1)
        # y_2 = torch.einsum("nik,iok->no", x_2_all, self.weight_2)

        y_0 = x_k_to_0 + self.biases_0
        y_1 = x_k_to_1 + self.biases_1
        y_2 = x_k_to_2 + self.biases_2

        if self.update_func is None:
            return y_0, y_1, y_2
        return self.update(y_0), self.update(y_1), self.update(y_2)
