from abc import ABC, abstractmethod
import torch_geometric
from omegaconf import DictConfig


class AbstractLoader(ABC):
    """
    Abstract class that provides an interface to load data.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        self.cfg = parameters

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameters={self.cfg})"

    @abstractmethod
    def load(self) -> torch_geometric.data.Data:
        """
        Load data into Data.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError
