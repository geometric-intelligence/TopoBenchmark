import torch
import torch_geometric
from abc import abstractmethod

class AbstractHeadModel(torch.nn.Module):
    r"""Head model.

    Parameters
    ----------
    in_channels: int
        Input dimension.
    out_channels: int
        Output dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        
    ):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
    
    def __call__(self, model_out: dict, batch: torch_geometric.data.Data) -> dict:
        x = self.forward(model_out, batch)
        model_out["logits"] = self.linear(x)
        return model_out
    
    @abstractmethod
    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Forward pass.

        Parameters
        ----------
        model_out: dict
            Dictionary containing the model output.
        batch: torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        x: torch.Tensor
            Output tensor over which the final linear layer is applied.
        """
        pass
        