"""KAN based GCN model."""

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_remaining_self_loops,
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value

from topobenchmarkx.nn.modules import KAN


def gcn_norm(
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: int | None = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: torch.dtype | None = None,
):
    """
    Function to normalize the adjacency matrix for the GCN layer.

    Parameters
    ----------
    edge_index : torch_geometric.typing.Adj
        Edge indices.
    edge_weight : torch.Tensor, optional
        Edge weights.
    num_nodes : int, optional
        Number of nodes.
    improved : bool, optional
        If set to :obj:`True`, the layer computes :math:`mathbf{hat{A}}` as :math:`mathbf{A} + 2mathbf{I}`.
        (default: :obj:`False`).
    add_self_loops : bool, optional
        If set to :obj:`False`, will not add self-loops to the input graph. By default, self-loops will be added
        in case :obj:`normalize` is set to :obj:`True`, and not added otherwise. (default: :obj:`None`).
    flow : str, optional
        The flow direction of message passing. Must be one of :obj:`"source_to_target"` or :obj:`"target_to_source"`.
        (default: :obj:`"source_to_target"`).
    dtype : torch.dtype, optional
        Data type to use for computation. If set to :obj:`None`, the data type of :obj:`edge_weight` will be used.

    Returns
    -------
    torch_geometric.typing.OptPairTensor
        The normalized edge indices and weights.
    """
    fill_value = 2.0 if improved else 1.0

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.0, dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError(
                "Sparse CSC matrices are not yet " "supported in 'gcn_norm'"
            )

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce="sum")
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ["source_to_target", "target_to_source"]
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes
        )

    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == "source_to_target" else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce="sum")
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


class KANGCNConv(MessagePassing):
    """
    KAN version of the graph convolutional operator.

    From the "Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907> paper.

    The layer computes the following operation:

    .. math::
        mathbf{X}^{prime} = mathbf{hat{D}}^{-1/2} mathbf{hat{A}}
        mathbf{hat{D}}^{-1/2} mathbf{X} mathbf{Theta},

    where :math:`mathbf{hat{A}} = mathbf{A} + mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`hat{D}_{ii} = sum_{j=0} hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        mathbf{x}^{prime}_i = mathbf{Theta}^{top} sum_{j in
        mathcal{N}(i) cup { i }} frac{e_{j,i}}{sqrt{hat{d}_j
        hat{d}_i}} mathbf{x}_j

    with :math:`hat{d}_i = 1 + sum_{j in mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`).

    Parameters
    ----------
    in_channels : int
        Size of each input sample, or :obj:`-1` to derive the size from the first input(s) to the forward method.
    out_channels : int
        Size of each output sample.
    kan_params : dict
        Parameters for the KAN layer.
    improved : bool, optional
        If set to :obj:`True`, the layer computes :math:`mathbf{hat{A}}` as :math:`mathbf{A} + 2mathbf{I}`.
        (default: :obj:`False`).
    cached : bool, optional
        If set to :obj:`True`, the layer will cache the computation of :math:`mathbf{hat{D}}^{-1/2} mathbf{hat{A}}
        mathbf{hat{D}}^{-1/2}` on first execution, and will use the cached version for further executions.
        This parameter should only be set to :obj:`True` in transductive learning scenarios. (default: :obj:`False`).
    add_self_loops : bool, optional
        If set to :obj:`False`, will not add self-loops to the input graph. By default, self-loops will be added
        in case :obj:`normalize` is set to :obj:`True`, and not added otherwise. (default: :obj:`None`).
    normalize : bool, optional
        Whether to add self-loops and compute symmetric normalization coefficients on-the-fly.
        (default: :obj:`True`).
    bias : bool, optional
        If set to :obj:`False`, the layer will not learn an additive bias. (default: :obj:`True`).
    **kwargs : optional
        Additional arguments of :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: OptPairTensor | None
    _cached_adj_t: SparseTensor | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kan_params: dict,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool | None = None,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(
                f"'{self.__class__.__name__}' does not support "
                f"adding self-loops to the graph when no "
                f"on-the-fly normalization is applied"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = KAN(
            in_channels,
            out_channels,
            **kan_params,
        )  # Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        super().reset_parameters()
        # self.lin.reset_parameters()
        # zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(
        self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix.
        edge_index : torch_geometric.typing.Adj
            Graph connectivity in COO format with shape [2, num_edges].
        edge_weight : torch.Tensor, optional
            Edge weight vector with shape [num_edges].

        Returns
        -------
        torch.Tensor
            The output node features.
        """
        if isinstance(x, (tuple, list)):
            raise ValueError(
                f"'{self.__class__.__name__}' received a tuple "
                f"of node features as input while this layer "
                f"does not support bipartite message passing. "
                f"Please try other layers such as 'SAGEConv' or "
                f"'GraphConv' instead"
            )

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        self.flow,
                        x.dtype,
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        self.flow,
                        x.dtype,
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        """
        Construct messages to node i in analogy to the GCN layer.

        Parameters
        ----------
        x_j : torch.Tensor
            Input features of neighboring nodes.
        edge_weight : torch.Tensor
            Edge weights.

        Returns
        -------
        torch.Tensor
            The constructed messages.
        """
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        """
        Message and aggregation function.

        Parameters
        ----------
        adj_t : torch_geometric.typing.Adj
            Adjacency matrix.
        x : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            The aggregated messages.
        """
        return spmm(adj_t, x, reduce=self.aggr)
