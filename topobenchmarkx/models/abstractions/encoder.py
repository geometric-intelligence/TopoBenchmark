from abc import ABC, abstractmethod

import torch
import torch_geometric


class AbstractInitFeaturesEncoder(torch.nn.Module):
    """abstract class that provides an interface to define a custom initial feature encoders"""

    def __init__(self):
        return

    @abstractmethod
    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Forward pass of the feature encoder model

        Parameters:
            :data: torch_geometric.data.Data

        Returns:
            :data: torch_geometric.data.Data

        """
