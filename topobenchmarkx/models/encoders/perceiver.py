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
    
    Parameters
    ----------
    dim: int
        Size of the dimension to normalize.
    fn: torch.nn.Module
        Function after LayerNorm.
    context_dim: int
        Size of the context to normalize.
    """
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        r"""Forward pass.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        kwargs: dict
            Dictionary of keyword arguments.
        
        Returns
        -------
        torch.Tensor
            Output tensor.
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
        r"""Forward pass.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        """
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    r"""Feedforward network.
    
    Parameters
    ----------
    dim: int
        Size of the input dimension.
    mult: int
        Multiplier for the hidden dimension.
    """
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        r"""Forward pass.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        """
        return self.net(x)


class Attention(nn.Module):
    r"""Attention function.
    
    Parameters
    ----------
    query_dim: int
        Size of the query dimension.
    context_dim: int
        Size of the context dimension.
    heads: int
        Number of heads.
    dim_head: int
        Size for each head.
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

    def forward(self, x, context=None, mask=None):
        r"""Forward pass.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        context: torch.Tensor
            Context tensor.
        mask: torch.Tensor
            Mask for attention calculation purposes.
        
        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

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
    r"""Perceiver model.
    
    Parameters
    ----------
    depth: int
        Number of layers to add to the model.
    dim: int
        Size of the input dimension.
    num_latents: int
        Number of latent vectors.
    cross_heads: int
        Number of heads for cross attention.
    latent_heads: int
        Number of heads for latent attention.
    cross_dim_head: int
        Size of the cross attention head.
    latent_dim_head: int
        Size of the latent attention head.
    weight_tie_layers: bool
        Whether to tie the weights of the layers.
    decoder_ff: bool
        Whether to use a feedforward network in the decoder.
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
                        latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head
                    ),
                    context_dim=dim,
                ),
                PreNorm(latent_dim, FeedForward(latent_dim)),
            ]
        )

        get_latent_attn = lambda: PreNorm(
            latent_dim,
            Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head),
        )
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                )
            )

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(
                queries_dim, latent_dim, heads=cross_heads, dim_head=cross_dim_head
            ),
            context_dim=latent_dim,
        )
        self.decoder_ff = (
            PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        )

        # self.to_logits = (
        #     nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()
        # )

    def forward(self, data, mask=None, queries=None):
        r"""Forward pass.
        
        Parameters
        ----------
        data: torch.Tensor
            Input tensor.
        mask: torch.Tensor
            Mask for attention calculation purposes.
        queries: torch.Tensor
            Queries tensor.
        """
        b, *_, device = *data.shape, data.device

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
