import torch
import torch_geometric
from abc import abstractmethod
from torch_geometric.utils import scatter

class AbstractZeroCellReadOut(torch.nn.Module):
    r"""Readout layer for GNNs that operates on the batch level.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        task_level: str,
        pooling_type: str = "sum",
        **kwargs,
    ):
        super().__init__()
        
        self.linear = torch.nn.Linear(hidden_dim, out_channels)
        assert task_level in ["graph", "node"], "Invalid task_level"
        self.task_level = task_level

        assert pooling_type in ["max", "sum", "mean"], "Invalid pooling_type"
        self.pooling_type = pooling_type
        
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __call__(self, model_out: dict, batch: torch_geometric.data.Data) -> dict:
        """Readout logic based on model_output."""
        # Readout
        model_out = self.forward(model_out, batch)

        model_out["logits"] = self.compute_logits(model_out["x_0"], batch["batch_0"])
         
        return model_out
    
    def compute_logits(self, x, batch):
        if self.task_level == "graph":
            if self.pooling_type == "max":
                x = scatter(x, batch, dim=0, reduce="max")

            elif self.pooling_type == "mean":
                x = scatter(x, batch, dim=0, reduce="mean")

            elif self.pooling_type == "sum":
                x = scatter(x, batch, dim=0, reduce="sum") 
        return self.linear(x)
        
    @abstractmethod
    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Forward pass.

        Args:
            model_out (dict): Dictionary containing the model output.
            batch (torch_geometric.data.Data): Batch object containing the batched domain data.
        Returns:
            dict: Dictionary containing the updated model output.
        """

