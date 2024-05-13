import torch
from torch_geometric.utils import scatter


class DefaultHead(torch.nn.Module):
    r"""Head model.

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
        in_channels: int,
        out_channels: int,
        task_level: str,
        pooling_type: str = "sum",
    ):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

        assert task_level in ["graph", "node"], "Invalid task_level"
        self.task_level = task_level

        assert pooling_type in ["max", "sum", "mean"], "Invalid pooling_type"
        self.pooling_type = pooling_type

    def forward(self, model_out: dict):
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
        x = model_out["x_0"]
        batch = model_out["batch_0"]
        if self.task_level == "graph":
            if self.pooling_type == "max":
                x = scatter(x, batch, dim=0, reduce="max")

            elif self.pooling_type == "mean":
                x = scatter(x, batch, dim=0, reduce="mean")

            elif self.pooling_type == "sum":
                x = scatter(x, batch, dim=0, reduce="sum")

        model_out["logits"] = self.linear(x)
        return model_out
