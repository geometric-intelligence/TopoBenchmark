import torch

from topobenchmarkx.models.readout import AbstractReadOut

class NodeLevelReadOut(AbstractReadOut):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task_level: str,
        pooling_type: str = "sum",
    ):
        super(AbstractReadOut, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

        assert task_level in ["graph", "node"], "Invalid task_level"
        self.task_level = task_level

        assert pooling_type in ["max", "sum", "mean"], "Invalid pooling_type"
        self.pooling_type = pooling_type

    def forward(self, model_out: dict):
        x = model_out["x_0"]
        if self.task_level == "graph":
            if self.pooling_type == "max":
                x = torch.max(x, dim=0)[0]

            elif self.pooling_type == "mean":
                x = torch.mean(x, dim=0)[0]

            elif self.pooling_type == "sum":
                x = torch.sum(x, dim=0)[0]

        model_out["logits"] = self.linear(x)
        return model_out
