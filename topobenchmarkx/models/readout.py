import torch


class ReadOut(torch.nn.Module):
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

    def forward(self, x: torch.Tensor):
        if self.task_level == "graph":
            if self.pooling_type == "max":
                x = torch.max(x, dim=0)[0]

            elif self.pooling_type == "mean":
                x = torch.mean(x, dim=0)[0]

            elif self.pooling_type == "sum":
                x = torch.sum(x, dim=0)[0]

        return self.linear(x)
