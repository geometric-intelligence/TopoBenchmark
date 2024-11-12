"""Loader for manually loaded graph datasets."""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from topobenchmarkx.data.loaders.base import AbstractLoader
from topobenchmarkx.data.utils import load_manual_graph
from topobenchmarkx.dataloader import DataloadDataset


class ManualGraphDatasetLoader(AbstractLoader):
    """Load manually provided graph datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_name: Name of the dataset
            - data_dir: Root directory for data
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        # You can add any specific validation logic for manual graphs if needed
        if not self.parameters.data_name:
            raise ValueError("The 'data_name' parameter must be provided.")

    def load_dataset(self) -> Any:
        """Load the manual graph dataset.

        Returns
        -------
        DataloadDataset
            The dataset object containing the manually loaded graph.
        """

        # Load the graph data using the manual graph loader function
        data = load_manual_graph()

        # Create and return the dataset object
        dataset = DataloadDataset([data])
        return dataset

    def get_data_dir(self) -> Path:
        """Get the data directory.

        Returns
        -------
        Path
            The path to the dataset directory.
        """
        return Path(
            self.parameters.data_dir
        )  # Assuming 'data_dir' is in the config
