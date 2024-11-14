"""Init file for custom loss module."""

from .DatasetLoss import DatasetLoss
from .GraphMLPLoss import GraphMLPLoss

__all__ = [
    "GraphMLPLoss",
    "DatasetLoss",
]
