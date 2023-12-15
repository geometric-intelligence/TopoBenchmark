from abc import ABC, abstractmethod

import torch_geometric


class AbstractLifting(torch_geometric.transforms.BaseTransform):
    """abstract class that provides an interface to define a custom readout"""

    def __init__(self):
        self.cache = {}
        return

    @abstractmethod
    def forward(self, batch: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
        """Forward pass of the lifting
        """
        
    def __call__(self, batch: torch_geometric.data.Batch, batch_idx):
        if batch_idx in self.cache:
            return self.cache[batch_idx]
        
        lifted = self.forward(batch)
        self.cache[batch_idx] = lifted
        return lifted
