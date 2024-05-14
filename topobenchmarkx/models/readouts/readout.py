import torch
import torch_geometric

from abc import abstractmethod

class AbstractReadOut(torch.nn.Module):
    r"""Readout layer for GNNs that operates on the batch level.

    Parameters
    ----------
    in_channels: int
        Input dimension.
    out_channels: int
        Output dimension.
    task_level: str
        Task level, either "graph" or "node". If "graph", the readout layer will pool the node embeddings to the graph level to obtain a single graph embedding for each batched graph. If "node", the readout layer will return the node embeddings.
    pooling_type: str
        Pooling type, either "max", "sum", or "mean". Specifies the type of pooling operation to be used for the graph-level embedding.
    """

    def __init__(self,):
        super().__init__()

    def __call__(self, model_out: dict, batch: torch_geometric.data.Data) -> dict:
        """Readout logic based on model_output."""
        return self.forward(model_out, batch)
    
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
        dict
            Dictionary containing the updated model output. Resulting key is "logits".
        """