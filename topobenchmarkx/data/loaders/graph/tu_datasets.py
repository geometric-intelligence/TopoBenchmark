"""Loaders for TU datasets."""

import os
from pathlib import Path
from typing import ClassVar

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset

from topobenchmarkx.data.loaders.base import AbstractLoader


class TUDatasetLoader(AbstractLoader):
    """Load TU datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - data_type: Type of the dataset (e.g., "graph_classification")
    """

    VALID_DATASETS: ClassVar[set[str]] = {
        "MUTAG",
        "ENZYMES",
        "PROTEINS",
        "COLLAB",
        "IMDB-BINARY",
        "IMDB-MULTI",
        "REDDIT-BINARY",
        "NCI1",
        "NCI109",
    }
    VALID_TYPES: ClassVar[set[str]] = {"TUDataset"}

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate the input parameters."""
        if self.parameters.data_name not in self.VALID_DATASETS:
            raise ValueError(
                f"Dataset '{self.parameters.data_name}' not supported. "
                f"Must be one of: {', '.join(sorted(self.VALID_DATASETS))}"
            )

        if self.parameters.data_type not in self.VALID_TYPES:
            raise ValueError(
                f"Data type '{self.parameters.data_type}' not supported. "
                f"Must be one of: {', '.join(sorted(self.VALID_TYPES))}"
            )

    def load_dataset(self) -> Dataset:
        """Load TU dataset.

        Returns
        -------
        Dataset
            The loaded TU dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = TUDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            use_node_attr=False,
        )
        return dataset

    def get_data_dir(self) -> Path:
        """Get the data directory.

        Returns
        -------
        Path
            The path to the dataset directory.
        """
        return os.path.join(self.root_data_dir, self.parameters.data_name)
