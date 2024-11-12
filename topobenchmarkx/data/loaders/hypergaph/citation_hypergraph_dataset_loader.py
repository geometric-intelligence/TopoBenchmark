"""Loaders for Citation Hypergraph dataset."""

import os
from pathlib import Path
from typing import ClassVar

from omegaconf import DictConfig

from topobenchmarkx.data.datasets import CitationHypergraphDataset
from topobenchmarkx.data.loaders.base import AbstractLoader


class CitationHypergraphDatasetLoader(AbstractLoader):
    """Load Citation Hypergraph dataset with configurable parameters.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - other relevant parameters
    """

    VALID_DATASETS: ClassVar[set] = {
        "cocitation_cora",
        "cocitation_citeseer",
        "coauthorship_cora",
        "coauthorship_dblp",
        "cocitation_pubmed",
    }
    VALID_TYPES: ClassVar[set] = {"coauthorship"}

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

    def load_dataset(self) -> CitationHypergraphDataset:
        """Load the Citation Hypergraph dataset.

        Returns
        -------
        CitationHypergraphDataset
            The loaded Citation Hypergraph dataset with the appropriate `data_dir`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = self._initialize_dataset()
        self.data_dir = self.get_data_dir()
        return dataset

    def _initialize_dataset(self) -> CitationHypergraphDataset:
        """Initialize the Citation Hypergraph dataset.

        Returns
        -------
        CitationHypergraphDataset
            The initialized dataset instance.
        """
        return CitationHypergraphDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
        )

    def get_data_dir(self) -> Path:
        """Get the data directory.

        Returns
        -------
        Path
            The path to the dataset directory.
        """
        return os.path.join(self.root_data_dir, self.parameters.data_name)
