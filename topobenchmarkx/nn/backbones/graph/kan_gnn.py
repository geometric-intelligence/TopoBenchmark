import copy
import inspect
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Linear, ModuleList
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import CachedLoader, NeighborLoader
from torch_geometric.nn.conv import (
    EdgeConv,
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    MessagePassing,
    PNAConv,
    SAGEConv,
)
from torch_geometric.nn.models import MLP
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils._trim_to_layer import TrimToLayer

from topobenchmarkx.nn.modules import KANGCNConv


class KANBasicGNN(torch.nn.Module):
    r"""An abstract class for implementing basic KAN-based GNN models, adapted from torch_geometric.nn.models.BasicGNN.

    Parameters
    ----------
    in_channels : int or tuple
        Size of each input sample, or `-1` to derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target dimensionalities.
    hidden_channels : int
        Size of each hidden sample.
    num_layers : int
        Number of message passing layers.
    kan_model : str, optional
        KAN model to use (default: "original").
    kan_params : dict
        Parameters for the KAN layer.
    out_channels : int, optional
        If not set to `None`, will apply a final linear transformation to convert hidden node embeddings to output size `out_channels`. Default is `None`.
    norm : str or Callable, optional
        The normalization function to use. Default is `None`.
    norm_kwargs : dict, optional
        Arguments passed to the respective normalization function defined by `norm`. Default is `None`.
    **kwargs : optional
        Additional arguments of the underlying `torch_geometric.nn.conv.MessagePassing` layers.

    Attributes
    ----------
    supports_edge_weight : bool
        Indicates if the model supports edge weights.
    supports_edge_attr : bool
        Indicates if the model supports edge attributes.
    supports_norm_batch : bool
        Indicates if the model supports batch normalization.

    Methods
    -------
    init_conv(in_channels, out_channels, **kwargs)
        Initializes the convolutional layer. Must be implemented by subclasses.
    reset_parameters()
        Resets all learnable parameters of the module.
    forward(x, edge_index, edge_weight=None, edge_attr=None, batch=None, batch_size=None, num_sampled_nodes_per_hop=None, num_sampled_edges_per_hop=None)
        Forward pass.
    inference_per_layer(layer, x, edge_index, batch_size)
        Performs inference for a specific layer.
    inference(loader, device=None, embedding_device='cpu', progress_bar=False, cache=False)
        Performs layer-wise inference on large-graphs using a `torch_geometric.loader.NeighborLoader`.

    Notes
    -----
    The `forward` method supports various input arguments for different scenarios, including edge weights, edge attributes, and batch information.
    """
    supports_edge_weight: Final[bool]
    supports_edge_attr: Final[bool]
    supports_norm_batch: Final[bool]

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        kan_model: str = "original",
        kan_params: dict = {},
        out_channels: Optional[int] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.norm = norm if isinstance(norm, str) else None
        self.norm_kwargs = norm_kwargs

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = ModuleList()
        if num_layers > 1:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, kan_model=kan_model, kan_params=kan_params, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, kan_model=kan_model, kan_params=kan_params, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels
        if out_channels is not None:
            self._is_conv_to_out = True
            self.convs.append(
                self.init_conv(in_channels, out_channels, kan_model=kan_model, kan_params=kan_params, **kwargs))
        else:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, kan_model=kan_model, kan_params=kan_params, **kwargs))

        self.norms = ModuleList()
        norm_layer = normalization_resolver(
            norm,
            hidden_channels,
            **(norm_kwargs or {}),
        )
        if norm_layer is None:
            norm_layer = torch.nn.Identity()

        self.supports_norm_batch = False
        if hasattr(norm_layer, 'forward'):
            norm_params = inspect.signature(norm_layer.forward).parameters
            self.supports_norm_batch = 'batch' in norm_params

        for _ in range(num_layers - 1):
            self.norms.append(copy.deepcopy(norm_layer))

        self.norms.append(torch.nn.Identity())

        # We define `trim_to_layer` functionality as a module such that we can
        # still use `to_hetero` on-top.
        self._trim = TrimToLayer()

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        raise NotImplementedError

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
    ) -> Tensor:
        r"""Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input node features.
        edge_index : torch.Tensor or SparseTensor
            The edge indices.
        edge_weight : torch.Tensor, optional
            The edge weights (if supported by the underlying GNN layer). Default is None.
        edge_attr : torch.Tensor, optional
            The edge features (if supported by the underlying GNN layer). Default is None.
        batch : torch.Tensor, optional
            The batch vector :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each element to a specific example. Only needs to be passed in case the underlying
            normalization layers require the `batch` information. Default is None.
        batch_size : int, optional
            The number of examples :math:`B`. Automatically calculated if not given. Only needs
            to be passed in case the underlying normalization layers require the `batch` information.
            Default is None.
        num_sampled_nodes_per_hop : List[int], optional
            The number of sampled nodes per hop. Useful in `torch_geometric.loader.NeighborLoader`
            scenarios to only operate on minimal-sized representations. Default is None.
        num_sampled_edges_per_hop : List[int], optional
            The number of sampled edges per hop. Useful in `torch_geometric.loader.NeighborLoader`
            scenarios to only operate on minimal-sized representations. Default is None.

        Returns
        -------
        torch.Tensor
            The output node features.
        """
        if (num_sampled_nodes_per_hop is not None
                and isinstance(edge_weight, Tensor)
                and isinstance(edge_attr, Tensor)):
            raise NotImplementedError("'trim_to_layer' functionality does not "
                                      "yet support trimming of both "
                                      "'edge_weight' and 'edge_attr'")

        xs: List[Tensor] = []
        assert len(self.convs) == len(self.norms)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if (not torch.jit.is_scripting()
                    and num_sampled_nodes_per_hop is not None):
                x, edge_index, value = self._trim(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                if edge_weight is not None:
                    edge_weight = value
                else:
                    edge_attr = value

            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x = conv(x, edge_index, edge_weight=edge_weight,
                         edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = conv(x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)

            if i < self.num_layers - 1:
                if self.supports_norm_batch:
                    x = norm(x, batch, batch_size)
                else:
                    x = norm(x)

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')


class KANGCN(KANBasicGNN):
    """KAN-based GCN adapted from torch_geometric.nn.models.GCN.
    
    Based on the Graph Neural Network from the `"Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, using the :class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.

    Parameters
    ----------
    in_channels : int
        Size of each input sample, or `-1` to derive the size from the first input(s) to the forward method.
    hidden_channels : int
        Size of each hidden sample.
    num_layers : int
        Number of message passing layers.
    kan_params : dict
        Parameters for the KAN layer.
    kan_model : str, optional
        KAN model to use (default: "original").
    out_channels : int, optional
        If not set to `None`, will apply a final linear transformation to convert hidden node embeddings to output size `out_channels`. Default is `None`.
    norm : str or Callable, optional
        The normalization function to use. Default is `None`.
    norm_kwargs : dict, optional
        Arguments passed to the respective normalization function defined by `norm`. Default is `None`.
    **kwargs : optional
        Additional arguments of :class:`torch_geometric.nn.conv.GCNConv`.
    """
    supports_edge_weight: Final[bool] = True
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        r"""Initializes the convolutional layer.
        
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        **kwargs : optional
            Additional keyword arguments.
            
        Returns
        -------
        MessagePassing
            The convolutional layer.
        """
        return KANGCNConv(in_channels, out_channels, **kwargs)

