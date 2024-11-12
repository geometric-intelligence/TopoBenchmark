"""Data loaders."""

# import os

# import numpy as np
import torch_geometric

# from omegaconf import DictConfig
from topobenchmarkx.data.loaders import LOADERS


class DatasetFetcher:
    """Loader for graph datasets.

    Parameters
    ----------
    data_source : str
        The type of dataset loader to use.
    **kwargs : dict
        Additional keyword arguments. Must contain:

        - parameters : DictConfig
            Configuration parameters for the dataset loader.
    """

    def __init__(self, data_source, **kwargs):
        super().__init__()
        self.parameters = kwargs["parameters"]
        self.loader = LOADERS[data_source](self.parameters)

    def __repr__(self) -> str:
        """Return a string representation of the GraphLoader object.

        Returns
        -------
        str
            String representation of the GraphLoader object.
        """
        return f"{self.__class__.__name__}(parameters={self.parameters})"

    def load(self) -> tuple[torch_geometric.data.Dataset, str]:
        """Load graph dataset.

        Returns
        -------
        tuple[torch_geometric.data.Dataset, str]
            Tuple containing the loaded data and the data directory.
        """
        # Define the path to the data directory
        dataset = self.loader.load_dataset()
        data_dir = self.loader.get_data_dir()

        return dataset, data_dir
