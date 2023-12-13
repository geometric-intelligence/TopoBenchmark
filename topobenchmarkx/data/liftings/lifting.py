from abc import ABC, abstractmethod

import torch_geometric


class AbstractLifting(torch_geometric.transforms.BaseTransform):
    """abstract class that provides an interface to define a custom readout"""

    def __init__(self):
        return

    @abstractmethod
    def forward(self, data: torch_geometric.data.Data) -> dict:
        """Forward pass of the lifting

        Parameters:
            :data: torch_geometric dataset

        Returns:
            :lifted_data: Dictionary with the added lifting data

        """
