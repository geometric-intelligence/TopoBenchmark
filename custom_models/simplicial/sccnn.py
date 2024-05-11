"""SCCNN implementation for complex classification."""

import torch
from topomodelx.nn.simplicial.sccnn_layer import SCCNNLayer


class SCCNNCusctom(torch.nn.Module):
    """SCCNN implementation for complex classification.

    Note: In this task, we can consider the output on any order of simplices for the
    classification task, which of course can be amended by a readout layer.

    Parameters
    ----------
    in_channels_all: tuple of int
        Dimension of input features on (nodes, edges, faces).
    hidden_channels_all: tuple of int
        Dimension of features of hidden layers on (nodes, edges, faces).
    conv_order: int
        Order of convolutions, we consider the same order for all convolutions.
    sc_order: int
        Order of simplicial complex.
    aggr_norm: bool
        Whether to normalize the aggregation.
    update_func: str
        Update function for the simplicial complex convolution.
    n_layers: int
        Number of layers.

    """

    def __init__(
        self,
        in_channels_all,
        hidden_channels_all,
        conv_order,
        sc_order,
        aggr_norm=False,
        update_func=None,
        n_layers=2,
    ):
        super().__init__()
        # first layer
        # we use an MLP to map the features on simplices of different dimensions to the same dimension
        self.in_linear_0 = torch.nn.Linear(in_channels_all[0], hidden_channels_all[0])
        self.in_linear_1 = torch.nn.Linear(in_channels_all[1], hidden_channels_all[1])
        self.in_linear_2 = torch.nn.Linear(in_channels_all[2], hidden_channels_all[2])

        self.layers = torch.nn.ModuleList(
            SCCNNLayer(
                in_channels=hidden_channels_all,
                out_channels=hidden_channels_all,
                conv_order=conv_order,
                sc_order=sc_order,
                aggr_norm=aggr_norm,
                update_func=update_func,
            )
            for _ in range(n_layers)
        )

    def forward(self, x_all, laplacian_all, incidence_all):
        """Forward computation.

        Parameters
        ----------
        x_all : tuple of tensors
            Tuple of feature tensors (node, edge, face).
            Each entry shape = (n_simplices, channels).

        laplacian_all : tuple of tensors
            Tuple of Laplacian tensors (graph laplacian L0, down edge laplacian L1_d, upper edge laplacian L1_u, face laplacian L2).
            Each entry shape = (n_simplices,n_simplices).

        incidence_all : tuple of tensors
            Tuple of order 1 and 2 incidence matrices.
            Shape of B1 = [n_nodes, n_edges].
            Shape of B2 = [n_edges, n_faces].

        Returns
        -------
        x_all : tuple of tensors
            Tuple of final hidden state tensors (node, edge, face).
            Each entry shape = (n_simplices, channels).
        """
        x_0, x_1, x_2 = x_all
        in_x_0 = self.in_linear_0(x_0)
        in_x_1 = self.in_linear_1(x_1)
        in_x_2 = self.in_linear_2(x_2)

        # Forward through SCCNN
        x_all = (in_x_0, in_x_1, in_x_2)
        for layer in self.layers:
            x_all = layer(x_all, laplacian_all, incidence_all)

        return x_all


# Layer
"""Simplicial Complex Convolutional Neural Network Layer."""
import torch
from torch.nn.parameter import Parameter


class SCCNNLayer(torch.nn.Module):
    r"""Layer of a Simplicial Complex Convolutional Neural Network.

    Parameters
    ----------
    in_channels : tuple of int
        Dimensions of input features on nodes, edges, and triangles.
    out_channels : tuple of int
        Dimensions of output features on nodes, edges, and triangles.
    conv_order : int
        Convolution order of the simplicial filters.
        To avoid too many parameters, we consider them to be the same.
    sc_order : int
        SC order.
    aggr_norm : bool, default = False
        Whether to normalize the aggregated message by the neighborhood size.
    update_func : str, default = None
        Activation function used in aggregation layers.
    initialization : str, default = "xavier_normal"
        Weight initialization method.

    Examples
    --------
    Here we provide an example of pseudocode for SCCNN layer in an SC
    of order two
    input X_0: [n_nodes, in_channels]
    input X_1: [n_edges, in_channels]
    input X_2: [n_faces, in_channels]

    graph Laplacian L_0: [n_nodes, n_nodes]
    1-Lap_down L_1_down: [n_edges, n_edges]
    1-Lap_up L_1_up: [n_edges, n_edges]
    2-Lap L_2: [n_faces,n_faces]
    1-incidence B_1: [n_nodes, n_edges]
    2-incidence B_2: [n_edges, n_faces]

    conv_order: int, e.g., 2

    output Y_0: [n_nodes, out_channels]
    output Y_1: [n_edges, out_channels]
    output Y_2: [n_faces, out_channels]

    SCCNN layer looks like:

        Y_0 = torch.einsum(
        concat(
            X_0, L_0@X_0, L_0@L_0@X_0 ||
            B_1@X_1, B_1@L_1_down@X_1, B_1@L_1_down@L_1_down@X_1
        ), weight_0)
        Y_1 = torch.einsum(
        concat(
            B_1.T@X_1, B_1.T@L_0@X_0, B_1.T@L_0@L_0@X_0 ||
            X_1, L_1_down@X_1, L_1_down@L_1_down@X_1,
                L_1_up@X_1, L_1_up@L_1_up@X_1 ||
            B_2@X_2, B_2@L_2@X_2, B_2@L_2@L_2@X_2
        ), weight_1)
        Y_2 = torch.einsum(
        concat(
            X_2, L_2@X_2, L_2@L_2@X_2 ||
            B_2.T@X_1, B_2.T@L_1_up@X_1, B_2.T@L_1_up@L_1_up@X_1
        ), weight_2)
    where
        - weight_0, weight_2, weight_2 are the trainable parameters
        - weight_0: [out_channels, in_channels, total_order_0]
            - total_order_0 = 1+conv_order + 1+conv_order
        - weight_1: [out_channels, in_channels, total_order_1]
            - total_order_1 = 1+conv_order +
                              1+conv_order+conv_order +
                              1+conv_order
        - weight_2: [out_channels, in_channels, total_order_2]
            - total_order_2 = 1+conv_order + 1+conv_order
        - to implement Lap_down@Lap_down@X, we consider chebyshev method
            to avoid matrix@matrix computation

    References
    ----------
    .. [1] Papillon, Sanborn, Hajij, Miolane.
        Equations of topological neural networks (2023).
        https://github.com/awesome-tnns/awesome-tnns/
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_order,
        sc_order,
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

        self.conv_order = conv_order
        self.sc_order = sc_order

        self.aggr_norm = aggr_norm
        self.update_func = update_func
        self.initialization = initialization

        assert initialization in ["xavier_uniform", "xavier_normal"]
        assert self.conv_order > 0

        self.weight_0 = Parameter(
            torch.Tensor(
                self.in_channels_0, self.out_channels_0, 1 + conv_order + 1 + conv_order
            )
        )

        self.weight_1 = Parameter(
            torch.Tensor(
                self.in_channels_1,
                self.out_channels_1,
                6 * conv_order + 3,
            )
        )

        # determine the third dimensions of the weights
        # because when SC order is larger than 2, there are lower and upper
        # parts for L_2; otherwise, L_2 contains only the lower part

        if sc_order > 2:
            self.weight_2 = Parameter(
                torch.Tensor(
                    self.in_channels_2,
                    self.out_channels_2,
                    4 * conv_order
                    + 2,  # in the future for arbitrary sc_order we should have this 6*conv_order + 3,
                )
            )

        elif sc_order == 2:
            self.weight_2 = Parameter(
                torch.Tensor(
                    self.in_channels_2,
                    self.out_channels_2,
                    4 * conv_order + 2,
                )
            )

        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.414):
        r"""Reset learnable parameters.

        Parameters
        ----------
        gain : float
            Gain for the weight initialization.

        Notes
        -----
        This function will be called by subclasses of
        MessagePassing that have trainable weights.
        """
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.weight_0, gain=gain)
            torch.nn.init.xavier_uniform_(self.weight_1, gain=gain)
            torch.nn.init.xavier_uniform_(self.weight_2, gain=gain)
        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.weight_0, gain=gain)
            torch.nn.init.xavier_normal_(self.weight_1, gain=gain)
            torch.nn.init.xavier_normal_(self.weight_2, gain=gain)
        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def aggr_norm_func(self, conv_operator, x):
        r"""Perform aggregation normalization."""
        neighborhood_size = torch.sum(conv_operator.to_dense(), dim=1)
        neighborhood_size_inv = 1 / neighborhood_size
        neighborhood_size_inv[~(torch.isfinite(neighborhood_size_inv))] = 0

        x = torch.einsum("i,ij->ij ", neighborhood_size_inv, x)
        x[~torch.isfinite(x)] = 0
        return x

    def update(self, x):
        """Update embeddings on each cell (step 4).

        Parameters
        ----------
        x : torch.Tensor, shape = (n_target_cells, out_channels)
            Feature tensor.

        Returns
        -------
        torch.Tensor, shape = (n_target_cells, out_channels)
            Updated output features on target cells.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x)
        return None

    def chebyshev_conv(self, conv_operator, conv_order, x):
        r"""Perform Chebyshev convolution.

        Parameters
        ----------
        conv_operator : torch.sparse, shape = (n_simplices,n_simplices)
            Convolution operator e.g., the adjacency matrix, or the Hodge Laplacians.
        conv_order : int
            The order of the convolution.
        x : torch.Tensor, shape = (n_simplices,num_channels)
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Output tensor. x[:, :, k] = (conv_operator@....@conv_operator) @ x.
        """
        num_simplices, num_channels = x.shape
        X = torch.empty(size=(num_simplices, num_channels, conv_order)).to(x.device)

        if self.aggr_norm:
            X[:, :, 0] = torch.mm(conv_operator, x)
            X[:, :, 0] = self.aggr_norm_func(conv_operator, X[:, :, 0])
            for k in range(1, conv_order):
                X[:, :, k] = torch.mm(conv_operator, X[:, :, k - 1])
                X[:, :, k] = self.aggr_norm_func(conv_operator, X[:, :, k])
        else:
            X[:, :, 0] = torch.mm(conv_operator, x)
            for k in range(1, conv_order):
                X[:, :, k] = torch.mm(conv_operator, X[:, :, k - 1])
        return X

    def forward(self, x_all, laplacian_all, incidence_all):
        r"""Forward computation (see [1]_).

        .. math::
            \begin{align*}
            &🟥 \quad m_{y \rightarrow z}^{(0\rightarrow1)}  = B_1^T \cdot h_y^{t,(0)} \cdot \Theta^{t,(0 \rightarrow 1)}\\
            &🟧 $\quad m_{z}^{(0\rightarrow1)}  = \frac{1}\sum_{y \in \mathcal{B}(z)} m_{y \rightarrow z}^{(0\rightarrow1)} \qquad \text{where} \sum \text{represents a mean.}\\
            &🟥 $\quad m_{z \rightarrow x}^{(1 \rightarrow 0)} = B_1\odot att(m_{z \in \mathcal{C}(x)}^{(0\rightarrow1)}, h_x^{t,(0)}) \cdot m_z^{(0\rightarrow1)} \cdot \Theta^{t,(1 \rightarrow 0)}\\
            &🟧 $\quad m_x^{(1\rightarrow0)}  = \sum_{z \in \mathcal{C}(x)} m_{z \rightarrow x}^{(1\rightarrow0)} \qquad \text{where} \sum \text{represents a mean.}\\
            &🟩 \quad m_x^{(0)}  = m_x^{(1\rightarrow0)}\\
            &🟦 \quad h_x^{t+1, (0)} = \Theta^{t, \text{update}} \cdot (h_x^{t,(0)}||m_x^{(0)})+b^{t, \text{update}}\\
            \end{align*}

        Parameters
        ----------
        x_all : tuple of tensors, shape = (x_0,x_1,x_2)
            Tuple of input feature tensors:

            - x_0: torch.Tensor, shape = (n_nodes,in_channels_0),
            - x_1: torch.Tensor, shape = (n_edges,in_channels_1),
            - x_2: torch.Tensor, shape = (n_triangles,in_channels_2).

        laplacian_all: tuple of tensors, shape = (laplacian_0,laplacian_down_1,laplacian_up_1,laplacian_2)
            Tuple of laplacian tensors:

            - laplacian_0: torch.sparse, graph Laplacian,
            - laplacian_down_1: torch.sparse, the 1-Hodge laplacian (lower part),
            - laplacian_up_1: torch.sparse, the 1-hodge laplacian (upper part),
            - laplacian_2: torch.sparse, the 2-hodge laplacian.

        incidence_all : tuple of tensors, shape = (b1,b2)
            Tuple of incidence tensors:

            - b1: torch.sparse, shape = (n_nodes,n_edges), node-to-edge incidence matrix,
            - b2: torch.sparse, shape = (n_edges,n_triangles), edge-to-face incidence matrix.

        Returns
        -------
        y_0 : torch.Tensor
            Output features on nodes.
        y_1 : torch.Tensor
            Output features on edges.
        y_2 : torch.Tensor
            Output features on triangles.
        """
        x_0, x_1, x_2 = x_all

        if self.sc_order == 2:
            laplacian_0, laplacian_down_1, laplacian_up_1, laplacian_2 = laplacian_all
        elif self.sc_order > 2:
            (
                laplacian_0,
                laplacian_down_1,
                laplacian_up_1,
                laplacian_down_2,
                laplacian_up_2,
            ) = laplacian_all

        # num_nodes, num_edges, num_triangles = x_0.shape[0], x_1.shape[0], x_2.shape[0]

        b1, b2 = incidence_all

        # identity_0, identity_1, identity_2 = (
        #     torch.eye(num_nodes).to(x_0.device),
        #     torch.eye(num_edges).to(x_0.device),
        #     torch.eye(num_triangles).to(x_0.device),
        # )

        """
        Convolution in the node space
        """
        # -----------Logic to obtain update for 0-cells --------
        # x_identity_0 = torch.unsqueeze(identity_0 @ x_0, 2)
        # x_0_to_0 = self.chebyshev_conv(laplacian_0, self.conv_order, x_0)
        # x_0_to_0 = torch.cat((x_identity_0, x_0_to_0), 2)

        x_0_laplacian = self.chebyshev_conv(laplacian_0, self.conv_order, x_0)
        x_0_to_0 = torch.cat([x_0.unsqueeze(2), x_0_laplacian], dim=2)
        # -------------------

        # x_1_to_0 = torch.mm(b1, x_1)
        # x_1_to_0_identity = torch.unsqueeze(identity_0 @ x_1_to_0, 2)
        # x_1_to_0 = self.chebyshev_conv(laplacian_0, self.conv_order, x_1_to_0)
        # x_1_to_0 = torch.cat((x_1_to_0_identity, x_1_to_0), 2)

        x_1_to_0_upper = torch.mm(b1, x_1)
        x_1_to_0_laplacian = self.chebyshev_conv(
            laplacian_0, self.conv_order, x_1_to_0_upper
        )
        x_1_to_0 = torch.cat([x_1_to_0_upper.unsqueeze(2), x_1_to_0_laplacian], dim=2)
        # -------------------

        x_0_all = torch.cat((x_0_to_0, x_1_to_0), 2)

        # -------------------
        """
        Convolution in the edge space
        """

        # -----------Logic to obtain update for 1-cells --------
        # x_identity_1 = torch.unsqueeze(identity_1 @ x_1, 2)
        # x_1_down = self.chebyshev_conv(laplacian_down_1, self.conv_order, x_1)
        # x_1_up = self.chebyshev_conv(laplacian_up_1, self.conv_order, x_1)
        # x_1_to_1 = torch.cat((x_identity_1, x_1_down, x_1_up), 2)

        x_1_down = self.chebyshev_conv(laplacian_down_1, self.conv_order, x_1)
        x_1_up = self.chebyshev_conv(laplacian_down_1, self.conv_order, x_1)
        x_1_to_1 = torch.cat((x_1.unsqueeze(2), x_1_down, x_1_up), 2)

        # -------------------

        # x_0_to_1 = torch.mm(b1.T, x_0)
        # x_0_to_1_identity = torch.unsqueeze(identity_1 @ x_0_to_1, 2)
        # x_0_to_1 = self.chebyshev_conv(laplacian_down_1, self.conv_order, x_0_to_1)
        # x_0_to_1 = torch.cat((x_0_to_1_identity, x_0_to_1), 2)

        # Lower projection
        x_0_1_lower = torch.mm(b1.T, x_0)

        # Calculate lowwer chebyshev_conv
        x_0_1_down = self.chebyshev_conv(laplacian_down_1, self.conv_order, x_0_1_lower)

        # Calculate upper chebyshev_conv (Note: in case of signed incidence should be always zero)
        x_0_1_up = self.chebyshev_conv(laplacian_up_1, self.conv_order, x_0_1_lower)

        # Concatenate output of filters
        x_0_to_1 = torch.cat([x_0_1_lower.unsqueeze(2), x_0_1_down, x_0_1_up], dim=2)
        # -------------------

        # x_2_to_1 = torch.mm(b2, x_2)
        # x_2_to_1_identity = torch.unsqueeze(identity_1 @ x_2_to_1, 2)
        # x_2_to_1 = self.chebyshev_conv(laplacian_up_1, self.conv_order, x_2_to_1)
        # x_2_to_1 = torch.cat((x_2_to_1_identity, x_2_to_1), 2)

        x_2_1_upper = torch.mm(b2, x_2)

        # Calculate lowwer chebyshev_conv (Note: In case of signed incidence should be always zero)
        x_2_1_down = self.chebyshev_conv(laplacian_down_1, self.conv_order, x_2_1_upper)

        # Calculate upper chebyshev_conv
        x_2_1_up = self.chebyshev_conv(laplacian_up_1, self.conv_order, x_2_1_upper)

        x_2_to_1 = torch.cat([x_2_1_upper.unsqueeze(2), x_2_1_down, x_2_1_up], dim=2)

        # -------------------
        x_1_all = torch.cat((x_0_to_1, x_1_to_1, x_2_to_1), 2)

        """
        convolution in the face (triangle) space, depending on the SC order,
        the exact form maybe a little different
        """
        # -------------------Logic to obtain update for 2-cells --------
        # x_identity_2 = torch.unsqueeze(identity_2 @ x_2, 2)

        # if self.sc_order == 2:
        #     x_2 = self.chebyshev_conv(laplacian_2, self.conv_order, x_2)
        #     x_2_to_2 = torch.cat((x_identity_2, x_2), 2)
        # elif self.sc_order > 2:
        #     x_2_down = self.chebyshev_conv(laplacian_down_2, self.conv_order, x_2)
        #     x_2_up = self.chebyshev_conv(laplacian_up_2, self.conv_order, x_2)
        #     x_2_to_2 = torch.cat((x_identity_2, x_2_down, x_2_up), 2)
        x_2_down = self.chebyshev_conv(laplacian_down_2, self.conv_order, x_2)
        x_2_up = self.chebyshev_conv(laplacian_up_2, self.conv_order, x_2)
        x_2_to_2 = torch.cat((x_2.unsqueeze(2), x_2_down, x_2_up), 2)

        # -------------------

        # x_1_to_2 = torch.mm(b2.T, x_1)
        # x_1_to_2_identity = torch.unsqueeze(identity_2 @ x_1_to_2, 2)
        # if self.sc_order == 2:
        #     x_1_to_2 = self.chebyshev_conv(laplacian_2, self.conv_order, x_1_to_2)
        # elif self.sc_order > 2:
        #     x_1_to_2 = self.chebyshev_conv(laplacian_down_2, self.conv_order, x_1_to_2)
        # x_1_to_2 = torch.cat((x_1_to_2_identity, x_1_to_2), 2)

        x_1_2_lower = torch.mm(b2.T, x_1)
        x_1_2_down = self.chebyshev_conv(laplacian_down_2, self.conv_order, x_1_2_lower)
        x_1_2_down = self.chebyshev_conv(laplacian_up_2, self.conv_order, x_1_2_lower)

        x_1_to_2 = torch.cat([x_1_2_lower.unsqueeze(2), x_1_2_down, x_1_2_down], dim=2)

        # That is my code, but to execute this part we need to have simplices order of k+1 in this case order of 3
        # x_3_2_upper = x_1_to_2 = torch.mm(b2, x_3)
        # x_3_2_down = self.chebyshev_conv(laplacian_down_2, self.conv_order, x_3_2_upper)
        # x_3_2_up = self.chebyshev_conv(laplacian_up_2, self.conv_order, x_3_2_upper)

        # x_3_to_2 = torch.cat([x_3_2_upper.unsueeze(2), x_3_2_down, x_3_2_up], dim=2)

        # -------------------

        x_2_all = torch.cat([x_1_to_2, x_2_to_2], dim=2)
        # The final version of this model should have the following line
        # x_2_all = torch.cat([x_1_to_2, x_2_to_2, x_3_to_2], dim=2)

        # -------------------

        # Need to check that this einsums are correct
        y_0 = torch.einsum("nik,iok->no", x_0_all, self.weight_0)
        y_1 = torch.einsum("nik,iok->no", x_1_all, self.weight_1)
        y_2 = torch.einsum("nik,iok->no", x_2_all, self.weight_2)

        if self.update_func is None:
            return y_0, y_1, y_2

        return self.update(y_0), self.update(y_1), self.update(y_2)
