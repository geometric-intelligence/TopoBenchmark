"""Loaders for Citation Hypergraph dataset."""

from omegaconf import DictConfig

from topobenchmarkx.data.datasets import MantraDataset
from topobenchmarkx.data.loaders.base import AbstractLoader


class MantraSimplicialDatasetLoader(AbstractLoader):
    """Load Mantra dataset with configurable parameters.

    Note: for the simplicial datasets it is necessary to include DatasetLoader into the name of the class!

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - other relevant parameters
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> MantraDataset:
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

    def _initialize_dataset(self) -> MantraDataset:
        """Initialize the Citation Hypergraph dataset.

        Returns
        -------
        CitationHypergraphDataset
            The initialized dataset instance.
        """
        return MantraDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
        )
