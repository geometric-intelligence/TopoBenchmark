"""Loaders for H3.6M dataset."""

from pathlib import Path

from omegaconf import DictConfig

from topobenchmarkx.data.datasets import H36MDataset
from topobenchmarkx.data.loaders.base import AbstractLoader


class H36MDatasetLoader(AbstractLoader):
    """Load H3.6M dataset with configurable year and task variable.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - year: Year of the dataset (if applicable)
            - task_variable: Task variable for the dataset
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> H36MDataset:
        """Load the H36M dataset.

        Returns
        -------
        H36MDataset
            The loaded H3.6M dataset with the appropriate `data_dir`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir(dataset)
        return dataset

    def _initialize_dataset(self) -> H36MDataset:
        """Initialize the H3.6M dataset.

        Returns
        -------
        H36MDataset
            The initialized dataset instance.
        """
        return H36MDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
        )

    def _redefine_data_dir(self, dataset: H36MDataset) -> Path:
        """Redefine the data directory based on the chosen (year, task_variable) pair.

        Parameters
        ----------
        dataset : H36MDataset
            The dataset instance.

        Returns
        -------
        Path
            The redefined data directory path.
        """
        return dataset.processed_root
