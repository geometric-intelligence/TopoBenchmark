"""Init file for custom loss module."""

from .GraphMLPLoss import GraphMLPLoss
from .DGMLoss import DGMLoss

__all__ = [
    "GraphMLPLoss",
    "DGMLoss",
]
