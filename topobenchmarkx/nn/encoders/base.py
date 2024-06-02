from abc import abstractmethod

import torch
import torch_geometric


class AbstractFeatureEncoder(torch.nn.Module):
    r"""Abstract class that provides an interface to define a custom feature
    encoder.

    Args:
        kwargs: Additional arguments.
    """

    def __init__(self, **kwargs):
        super().__init__()
        return

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Forward pass of the feature encoder model.

        Args:
            data (torch_geometric.data.Data): Input data object which should contain x features.
        Returns:
            torch_geometric.data.Data: Output data object with updated x features.
        """
