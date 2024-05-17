import torch
import torch_geometric

from abc import abstractmethod

class AbstractReadOut(torch.nn.Module):
    r"""Readout layer for GNNs that operates on the batch level.
    """

    def __init__(self,):
        super().__init__()
        
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __call__(self, model_out: dict, batch: torch_geometric.data.Data) -> dict:
        """Readout logic based on model_output."""
        return self.forward(model_out, batch)
    
    @abstractmethod
    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Forward pass.

        Args:
            model_out (dict): Dictionary containing the model output.
            batch (torch_geometric.data.Data): Batch object containing the batched domain data.
        Returns:
            dict: Dictionary containing the updated model output.
        """