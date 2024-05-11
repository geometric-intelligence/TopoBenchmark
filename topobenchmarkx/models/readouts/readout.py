import torch
import torch_geometric
from torch_geometric.utils import scatter


from topobenchmarkx.models.readouts.readouts import PropagateSignalDown
# Implemented Poolings
READOUTS = {
    "PropagateSignalDown": PropagateSignalDown
}


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
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

        self.signal_readout = kwargs["readout_name"] != "None"
        if self.signal_readout:
            signal_readout_name = kwargs.get("readout_name")
            self.readout = READOUTS[signal_readout_name](**kwargs)

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Forward pass.
        
        Parameters
        ----------
        model_out: dict
            Dictionary containing the model output.
        
        Returns
        -------
        dict
            Dictionary containing the updated model output. Resulting key is "logits".
        """
        # Propagate signal
        if self.signal_readout:
            model_out = self.readout(model_out, batch)

        return model_out
    
    

    