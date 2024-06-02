from abc import ABC, abstractmethod

import torch_geometric


class AbstractLoss(ABC):
    r"""Abstract class for the loss class."""

    def __init__(
        self,
    ):
        super().__init__()

    def __call__(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        r"""Loss logic based on model_output."""
        return self.forward(model_out, batch)

    @abstractmethod
    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Forward pass."""
