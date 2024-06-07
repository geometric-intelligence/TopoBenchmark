"""Abstract class for the optimizer manager class."""

from abc import ABC, abstractmethod


class AbstractOptimizer(ABC):
    r"""Abstract class for the optimizer manager class."""

    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def configure_optimizer(self, model_parameters: dict):
        r"""Configure the optimizer and scheduler.

        Act as a wrapper.

        Parameters
        ----------
        model_parameters : dict
            The model parameters.
        """
