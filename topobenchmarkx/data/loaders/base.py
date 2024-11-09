"""Abstract Loader class."""

from pathlib import Path
from abc import ABC, abstractmethod

import torch_geometric
from omegaconf import DictConfig


class AbstractLoader(ABC):
    """Abstract class that provides an interface to load data.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        self.parameters = parameters
        self.root_data_dir = Path(parameters["data_dir"])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameters={self.parameters})"

    @abstractmethod
    def load_dataset(self) -> torch_geometric.data.Data:
        """Load data into Data.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def get_data_dir(self) -> Path:
        """Get the data directory."""
        raise NotImplementedError
