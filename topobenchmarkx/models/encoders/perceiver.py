# Code taken from https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_io.py
from functools import wraps

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


# structured dropout, more effective than traditional attention dropouts

# def dropout_seq(seq, mask, dropout):
#     b, n, *_, device = *seq.shape, seq.device
#     logits = torch.randn(b, n, device = device)

#     if exists(mask):
#         logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

#     keep_prob = 1. - dropout
#     num_keep = max(1,  int(keep_prob * n))
#     keep_indices = logits.topk(num_keep, dim = 1).indices

#     batch_indices = torch.arange(b, device = device)
#     batch_indices = rearrange(batch_indices, 'b -> b 1')

#     seq = seq[batch_indices, keep_indices]

#     if exists(mask):
#         seq_counts = mask.sum(dim = -1)
#         seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
#         keep_mask = torch.arange(num_keep, device = device) < rearrange(seq_keep_counts, 'b -> b 1')

#         mask = mask[batch_indices, keep_indices] & keep_mask

#     return seq, mask

# helper classes


class PreNorm(nn.Module):
    r"""Class to wrap together LayerNorm and a specified function.

    Args:
        dim (int): Size of the dimension to normalize.
        fn (torch.nn.Module): Function after LayerNorm.
        context_dim (int, optional): Size of the context to normalize. (default: None)
    """

    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = (
            nn.LayerNorm(context_dim) if exists(context_dim) else None
        )
    
    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.norm.normalized_shape[0]}, fn={self.fn}, context_dim={self.norm_context.normalized_shape[0] if exists(self.norm_context) else None})"
    
    def forward(self, x, **kwargs):
        r"""Forward pass of the PreNorm class.

        Args:
            x (torch.Tensor): Input tensor.
            **kwargs: Additional arguments. If context_dim is not None the context tensor should be passed.
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    r"""GEGLU activation function."""

    def forward(self, x):
        r"""Forward pass of the GEGLU activation function.

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    r"""Feedforward network with two linear layers and GEGLU activation function in between.

    Args:
        dim (int): Size of the input dimension.
        mult (int, optional): Multiplier for the hidden dimension. (default: 4)
    """
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim)
        )
    
    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.net[0].in_features}, mult={self.net[0].out_features // self.net[0].in_features})"
    
    def forward(self, x):
        r"""Forward pass of the FeedForward class.

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x)


class Attention(nn.Module):
    r"""Attention class to calculate the attention weights.

    Args:
        query_dim (int): Size of the query dimension.
        context_dim (int, optional): Size of the context dimension. (default: None)
        heads (int, optional): Number of heads. (default: 8)
        dim_head (int, optional): Size for each head. (default: 64)
    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def __repr__(self):
        return f"{self.__class__.__name__}(query_dim={self.to_q.in_features}, context_dim={self.to_kv.in_features // 2}, heads={self.heads}, dim_head={self.to_q.out_features // self.heads})"
    
    def forward(self, x, context=None, mask=None):
        r"""Forward pass of the Attention class.

        Args:
            x (torch.Tensor): Input tensor.
            context (torch.Tensor, optional): Context tensor. (default: None)
            mask (torch.Tensor, optional): Mask for attention calculation purposes. (default: None)
        Returns:
            torch.Tensor: Output tensor.
        """
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v)
        )

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


# main class


class Perceiver(nn.Module):
    r"""Perceiver model. For more information https://arxiv.org/abs/2103.03206.

    Args:
        depth (int): Number of layers to add to the model.
        dim (int): Size of the input dimension.
        num_latents (int, optional): Number of latent vectors. (default: 1)
        cross_heads (int, optional): Number of heads for cross attention. (default: 1)
        latent_heads (int, optional): Number of heads for latent attention. (default: 8)
        cross_dim_head (int, optional): Size of the cross attention head. (default: 64)
        latent_dim_head (int, optional): Size of the latent attention head. (default: 64)
        weight_tie_layers (bool, optional): Whether to tie the weights of the layers. (default: False)
        decoder_ff (bool, optional): Whether to use a feedforward network in the decoder. (default: False)
    """

    def __init__(
        self,
        *,
        depth,
        dim,
        # logits_dim=None,
        num_latents=1,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
        # seq_dropout_prob = 0.
    ):
        super().__init__()
        # self.seq_dropout_prob = seq_dropout_prob

        latent_dim = dim
        queries_dim = dim

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    latent_dim,
                    Attention(
                        latent_dim,
                        dim,
                        heads=cross_heads,
                        dim_head=cross_dim_head,
                    ),
                    context_dim=dim,
                ),
                PreNorm(latent_dim, FeedForward(latent_dim)),
            ]
        )

        def get_latent_attn():
            return PreNorm(
                latent_dim,
                Attention(
                    latent_dim, heads=latent_heads, dim_head=latent_dim_head
                ),
            )

        def get_latent_ff():
            return PreNorm(latent_dim, FeedForward(latent_dim))

        get_latent_attn, get_latent_ff = map(
            cache_fn, (get_latent_attn, get_latent_ff)
        )

        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        get_latent_attn(**cache_args),
                        get_latent_ff(**cache_args),
                    ]
                )
            )

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(
                queries_dim,
                latent_dim,
                heads=cross_heads,
                dim_head=cross_dim_head,
            ),
            context_dim=latent_dim,
        )
        self.decoder_ff = (
            PreNorm(queries_dim, FeedForward(queries_dim))
            if decoder_ff
            else None
        )

        self.dim = dim
        self.num_latents = num_latents
        self.cross_heads = cross_heads
        self.latent_heads = latent_heads
        self.cross_dim_head = cross_dim_head
        self.latent_dim_head = latent_dim_head
        self.weight_tie_layers = weight_tie_layers
        self.decoder_ff = decoder_ff
        
        # self.to_logits = (
        #     nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()
        # )
    
    def __repr__(self):
        return f"{self.__class__.__name__}(depth={len(self.layers)}, dim={self.dim}, num_latents={self.num_latents}, cross_heads={self.cross_heads}, latent_heads={self.latent_heads}, cross_dim_head={self.cross_dim_head}, latent_dim_head={self.latent_dim_head}, weight_tie_layers={self.weight_tie_layers}, decoder_ff={self.decoder_ff}"

    def forward(self, data, mask=None, queries=None):
        r"""Forward pass of the Perceiver model.

        Args:
            data (torch.Tensor): Input tensor.
            mask (torch.Tensor, optional): Mask for attention calculation purposes. (default: None)
            queries (torch.Tensor, optional): Queries tensor. (default: None)
        Returns:
            torch.Tensor: Output tensor.
        """
        b, *_ = *data.shape

        x = repeat(self.latents, "n d -> b n d", b=b)

        cross_attn, cross_ff = self.cross_attend_blocks

        # structured dropout (as done in perceiver AR https://arxiv.org/abs/2202.07765)

        # if self.training and self.seq_dropout_prob > 0.:
        # data, mask = dropout_seq(data, mask, self.seq_dropout_prob)

        # cross attention only happens once for Perceiver IO

        x = cross_attn(x, context=data, mask=mask) + x
        x = cross_ff(x) + x

        # layers

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        if not exists(queries):
            return x.squeeze(1)

        # make sure queries contains batch dimension

        # if queries.ndim == 2:
        #     queries = repeat(queries, 'n d -> b n d', b = b)

        # cross attend from decoder queries to latents

        # latents = self.decoder_cross_attn(queries, context = x)

        # optional decoder feedforward

        # if exists(self.decoder_ff):
        #     latents = latents + self.decoder_ff(latents)

        # final linear out

        # return x #self.to_logits(latents)
        return None



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
