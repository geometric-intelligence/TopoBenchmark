"""Init file for graph load module."""

from .planetoid_datasets import PlanetoidDatasetLoader

GRAPH_LOADERS = {
    "planetoid_datasets": PlanetoidDatasetLoader,
}

__all__ = ["GRAPH_LOADERS"]
