import torch_geometric
from abc import ABC, abstractmethod

class AbstractltLoss(ABC):
    """Abstract class that provides an interface to loss logic within
    netowrk."""

    def __init__(self,):
        super().__init__()

    def __call__(self, model_out: dict, batch: torch_geometric.data.Data) -> dict:
        """Loss logic based on model_output."""
        return self.forward(model_out, batch)
    
    @abstractmethod
    def forward(self, model_out: dict, batch: torch_geometric.data.Data): 
        pass
