"""Graph backbone."""

from torch_geometric.nn.models import (
    GAT,
    GCN,
    GIN,
    MLP,
    PNA,
    DeepGraphInfomax,
    EdgeCNN,
    GraphSAGE,
    MetaLayer,
    Node2Vec,
)

from .identity_gnn import (
    IdentityGAT,
    IdentityGCN,
    IdentityGIN,
    IdentitySAGE,
)

from .kan_gnn import KANGCN

__all__ = [
    "IdentityGCN",
    "IdentityGIN",
    "IdentityGAT",
    "IdentitySAGE",
    "KANGCN",
    "MLP",
    "GCN",
    "GraphSAGE",
    "GIN",
    "GAT",
    "PNA",
    "EdgeCNN",
    "MetaLayer",
    "Node2Vec",
    "DeepGraphInfomax",
]
