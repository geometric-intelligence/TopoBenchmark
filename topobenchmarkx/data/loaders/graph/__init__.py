"""Init file for graph load module."""

from .hetero_datasets import HeterophilousGraphDatasetLoader
from .manual_graph_dataset_loader import ManualGraphDatasetLoader
from .modecule_datasets import MoleculeDatasetLoader
from .planetoid_datasets import PlanetoidDatasetLoader
from .tu_datasets import TUDatasetLoader
from .us_county_demos_dataset_loader import USCountyDemosDatasetLoader

GRAPH_LOADERS = {
    "PlanetoidDatasetLoader": PlanetoidDatasetLoader,
    "TUDatasetLoader": TUDatasetLoader,
    "HeterophilousGraphDatasetLoader": HeterophilousGraphDatasetLoader,
    "MoleculeDatasetLoader": MoleculeDatasetLoader,
    "USCountyDemosDatasetLoader": USCountyDemosDatasetLoader,
    "ManualGraphDatasetLoader": ManualGraphDatasetLoader,
}

__all__ = ["GRAPH_LOADERS"]
