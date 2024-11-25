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

from .graph_mlp import GraphMLP
from .identity_gnn import (
    IdentityGAT,
    IdentityGCN,
    IdentityGIN,
    IdentitySAGE,
)

__all__ = [
    "IdentityGCN",
    "IdentityGIN",
    "IdentityGAT",
    "IdentitySAGE",
    "GraphMLP",
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
