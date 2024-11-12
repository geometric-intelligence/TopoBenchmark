"""Loaders for US County Demos dataset."""

import os
from pathlib import Path
from typing import ClassVar

from omegaconf import DictConfig

from topobenchmarkx.data.datasets import USCountyDemosDataset
from topobenchmarkx.data.loaders.base import AbstractLoader


class USCountyDemosDatasetLoader(AbstractLoader):
    """Load US County Demos dataset with configurable year and task variable.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - year: Year of the dataset (if applicable)
            - task_variable: Task variable for the dataset
    """

    VALID_DATASETS: ClassVar[set] = {"US-county-demos"}
    VALID_TYPES: ClassVar[set] = {"USCountyDemosDataset"}

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

    def load_dataset(self) -> USCountyDemosDataset:
        """Load the US County Demos dataset.

        Returns
        -------
        USCountyDemosDataset
            The loaded US County Demos dataset with the appropriate `data_dir`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir(dataset)
        return dataset

    def _initialize_dataset(self) -> USCountyDemosDataset:
        """Initialize the US County Demos dataset.

        Returns
        -------
        USCountyDemosDataset
            The initialized dataset instance.
        """
        return USCountyDemosDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
        )

    def _redefine_data_dir(self, dataset: USCountyDemosDataset) -> Path:
        """Redefine the data directory based on the chosen (year, task_variable) pair.

        Parameters
        ----------
        dataset : USCountyDemosDataset
            The dataset instance.

        Returns
        -------
        Path
            The redefined data directory path.
        """
        return dataset.processed_root

    def get_data_dir(self) -> Path:
        """Get the data directory.

        Returns
        -------
        Path
            The path to the dataset directory.
        """
        return os.path.join(self.root_data_dir, self.parameters.data_name)
