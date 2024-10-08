"""Graph backbone."""

from .identity_gnn import (
    IdentityGAT,
    IdentityGCN,
    IdentityGIN,
    IdentitySAGE,
)

__all__ = ["IdentityGCN", "IdentityGIN", "IdentityGAT", "IdentitySAGE"]
