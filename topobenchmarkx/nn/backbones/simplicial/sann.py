"""SANN network."""

import torch
import torch.nn.functional
from torch.nn import ParameterList
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
        hop_num=3,
        n_layers=2,
    ):
        super().__init__()
        self.complex_dim = complex_dim
        self.hop_num = hop_num

        assert n_layers >= 1

        if isinstance(in_channels, int):  # If only one value is passed
            in_channels = [in_channels] * self.hop_num

        self.layers = torch.nn.ModuleList()

        # Set of simplices layers
        self.layers_0 = torch.nn.ModuleList(
            SANNLayer(
                [in_channels[i] for i in range(hop_num)],
                [hidden_channels] * hop_num,
                update_func=update_func,
                hop_num=hop_num,
            )
            for i in range(complex_dim)
        )
        self.layers.append(self.layers_0)

        # From layer 1 to n_layers
        for i in range(1, n_layers):
            self.layers.append(
                torch.nn.ModuleList(
                    SANNLayer(
                        [hidden_channels] * hop_num,
                        [hidden_channels] * hop_num,
                        update_func=update_func,
                        hop_num=hop_num,
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
    hop_num : int
        Number of hop representations to consider.
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
        hop_num,
        aggr_norm: bool = False,
        update_func=None,
        initialization: str = "xavier_normal",
    ) -> None:
        super().__init__()

        assert hop_num == len(
            in_channels
        ), "Number of hops must be equal to the number of input channels."
        assert hop_num == len(
            out_channels
        ), "Number of hops must be equal to the number of output channels."

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.aggr_norm = aggr_norm
        self.update_func = update_func
        self.initialization = initialization

        assert initialization in ["xavier_uniform", "xavier_normal"]

        self.weights = ParameterList(
            [
                Parameter(
                    torch.Tensor(
                        self.in_channels[i],
                        self.out_channels[i],
                    )
                )
                for i in range(hop_num)
            ]
        )
        self.biases = ParameterList(
            [
                Parameter(
                    torch.Tensor(
                        self.out_channels[i],
                    )
                )
                for i in range(hop_num)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.414):
        r"""Reset learnable parameters.

        Parameters
        ----------
        gain : float
            Gain for the weight initialization.
        """
        if self.initialization == "xavier_uniform":
            for i in range(len(self.weights)):
                torch.nn.init.xavier_uniform_(self.weights[i], gain=gain)
                torch.nn.init.zeros_(self.biases[i])
        elif self.initialization == "xavier_normal":
            for i in range(len(self.weights)):
                torch.nn.init.xavier_normal_(self.weights[i], gain=gain)
                torch.nn.init.zeros_(self.biases[i])
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
        t = len(x_all)
        x_k_t = {i: x_all[i] for i in range(t)}

        y_k_t = {
            i: torch.mm(x_k_t[i], self.weights[i]) + self.biases[i]
            for i in range(t)
        }

        # TODO Check aggregation as list of ys
        # Need to check that this einsums are correct
        # y_0 = torch.einsum("nik,iok->no", x_0_all, self.weight_0)
        # y_1 = torch.einsum("nik,iok->no", x_1_all, self.weight_1)
        # y_2 = torch.einsum("nik,iok->no", x_2_all, self.weight_2)

        if self.update_func is None:
            return tuple(y_k_t.values())

        return tuple([self.update(y_t) for y_t in y_k_t.values()])
