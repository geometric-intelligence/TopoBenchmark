from abc import abstractmethod

import torch
import torch_geometric


class AbstractFeatureEncoder(torch.nn.Module):
    """Abstract class that provides an interface to define a custom feature encoder."""

    def __init__(self, **kwargs):
        super().__init__()
        return

    def __call__(self, data):
        return self.forward(data)

    @abstractmethod
    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        """Forward pass of the feature encoder model.

        Parameters:
            :data: torch_geometric.data.Data

        Returns:
            :data: torch_geometric.data.Data
        """