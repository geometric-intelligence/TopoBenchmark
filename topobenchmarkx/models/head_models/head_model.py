import torch
import torch_geometric
from abc import abstractmethod

class AbstractHeadModel(torch.nn.Module):
    r"""Abstract head model class.

    Args:
        in_channels (int): Input dimension.
        out_channels (int): Output dimension.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        
    ):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        
    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.linear.in_features}, out_channels={self.linear.out_features})"
    
    def __call__(self, model_out: dict, batch: torch_geometric.data.Data) -> dict:
        x = self.forward(model_out, batch)
        model_out["logits"] = self.linear(x)
        return model_out
    
    @abstractmethod
    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Forward pass of the head model.

        Args:
            model_out (dict): Dictionary containing the model output.
            batch (torch_geometric.data.Data): Batch object containing the batched domain data.
        Returns:
            torch.Tensor: Output tensor over which the final linear layer is applied.
        """
        pass
        