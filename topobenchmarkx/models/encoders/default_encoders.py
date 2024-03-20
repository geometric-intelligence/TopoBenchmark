import torch
import torch_geometric
from torch_geometric.nn.norm import GraphNorm

from topobenchmarkx.models.abstractions.encoder import AbstractInitFeaturesEncoder


class BaseEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_channels, out_channels)
        self.linear2 = torch.nn.Linear(out_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.BN = GraphNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.BN(x, batch=batch) if batch.shape[0] > 0 else self.BN(x)
        x = self.dropout(self.relu(x))
        x = self.linear2(x)
        return x


class BaseFeatureEncoder(AbstractInitFeaturesEncoder):
    def __init__(
        self, in_channels, out_channels, proj_dropout=0, selected_dimensions=None
    ):
        super(AbstractInitFeaturesEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimensions = (
            selected_dimensions
            if selected_dimensions is not None
            else range(len(self.in_channels))
        )
        for i in self.dimensions:
            setattr(
                self,
                f"encoder_{i}",
                BaseEncoder(
                    self.in_channels[i], self.out_channels, dropout=proj_dropout
                ),
            )

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        if not hasattr(data, "x_0"):
            data.x_0 = data.x

        for i in self.dimensions:
            if hasattr(data, f"x_{i}") and hasattr(self, f"encoder_{i}"):
                batch = data.batch if i == 0 else getattr(data, f"batch_{i}")
                data[f"x_{i}"] = getattr(self, f"encoder_{i}")(data[f"x_{i}"], batch)
        return data


from topobenchmarkx.models.encoders.perceiver import Perceiver


class SetFeatureEncoder(AbstractInitFeaturesEncoder):
    def __init__(
        self, in_channels, out_channels, proj_dropout=0, selected_dimensions=None
    ):
        super(AbstractInitFeaturesEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimensions = (
            selected_dimensions
            if selected_dimensions is not None
            else range(len(self.in_channels))
        )
        for idx, i in enumerate(self.dimensions):
            if idx == 0:
                setattr(
                    self,
                    f"encoder_{i}",
                    BaseEncoder(
                        self.in_channels[i], self.out_channels, dropout=proj_dropout
                    ),
                )
            else:
                setattr(
                    self,
                    f"encoder_{i}",
                    Perceiver(
                        dim=self.out_channels,
                        depth=1,
                        cross_heads=4,
                        cross_dim_head=self.out_channels,
                        latent_dim_head=self.out_channels,
                    ),
                )

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        if not hasattr(data, "x_0"):
            data.x_0 = data.x

        for idx, i in enumerate(self.dimensions):
            if idx == 0:
                if hasattr(data, f"x_{i}") and hasattr(self, f"encoder_{i}"):
                    batch = data.batch if i == 0 else getattr(data, f"batch_{i}")
                    data[f"x_{i}"] = getattr(self, f"encoder_{i}")(
                        data[f"x_{i}"], batch
                    )
            else:
                if hasattr(data, f"x_{i}") and hasattr(self, f"encoder_{i}"):
                    cell_features = data["x_0"][data[f"x_{i}"].long()]
                    data[f"x_{i}"] = getattr(self, f"encoder_{i}")(cell_features)
                else:
                    data[f"x_{i}"] = torch.tensor([], device=data.x_0.device)
        return data
