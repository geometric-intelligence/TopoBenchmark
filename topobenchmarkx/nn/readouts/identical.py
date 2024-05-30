
import torch_geometric
from topobenchmarkx.nn.readouts.readout import AbstractZeroCellReadOut


class NoReadOut(AbstractZeroCellReadOut):
    r"""No readout layer. This readout layer does not perform any operation on the node embeddings."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, model_out: dict, batch: torch_geometric.data.Data) -> dict:
        r"""Forward pass of the no readout layer. It returns the model output without any modification.
        
        Args:
            model_out (dict): Dictionary containing the model output.
            batch (torch_geometric.data.Data): Batch object containing the batched domain data.
        Returns:
            model_out (dict): Dictionary containing the model output.
        """
        return model_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
