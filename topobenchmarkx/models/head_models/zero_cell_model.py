import torch
import torch_geometric
from torch_geometric.utils import scatter
from topobenchmarkx.models.head_models.head_model import AbstractHeadModel

class ZeroCellModel(AbstractHeadModel):
    r"""Zero cell head model. This model produces an output based only on the features of the nodes (the zero cells). The output is obtained by applying a linear layer to the input features. Based on the task level, the readout layer will pool the node embeddings to the graph level to obtain a single graph embedding for each batched graph or return a value for each node.

    Args:
        in_channels (int): Input dimension.
        out_channels (int): Output dimension.
        task_level (str): Task level, either "graph" or "node". If "graph", the readout layer will pool the node embeddings to the graph level to obtain a single graph embedding for each batched graph. If "node", the readout layer will return the node embeddings.
        pooling_type (str, optional): Pooling type, either "max", "sum", or "mean". Specifies the type of pooling operation to be used for the graph-level embedding. (default: "sum")
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task_level: str,
        pooling_type: str = "sum",
        **kwargs,
    ):
        super().__init__(in_channels, out_channels)

        assert task_level in ["graph", "node"], "Invalid task_level"
        self.task_level = task_level

        assert pooling_type in ["max", "sum", "mean"], "Invalid pooling_type"
        self.pooling_type = pooling_type
    
    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.linear.in_features}, out_channels={self.linear.out_features}, task_level={self.task_level}, pooling_type={self.pooling_type})"
   
    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Forward pass of the zero cell head model.

        Args:
            model_out (dict): Dictionary containing the model output.
            batch (torch_geometric.data.Data): Batch object containing the batched domain data.
        Returns:
            torch.Tensor: Output tensor.
        """
        x = model_out["x_0"]
        batch = batch["batch_0"]
        if self.task_level == "graph":
            if self.pooling_type == "max":
                x = scatter(x, batch, dim=0, reduce="max")

            elif self.pooling_type == "mean":
                x = scatter(x, batch, dim=0, reduce="mean")

            elif self.pooling_type == "sum":
                x = scatter(x, batch, dim=0, reduce="sum")

        return x