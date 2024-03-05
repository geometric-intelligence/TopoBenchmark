import torch
import torch_geometric
from torch_geometric.nn.norm import GraphNorm

from topobenchmarkx.models.abstractions.encoder import AbstractInitFeaturesEncoder


class BaseNodeFeatureEncoder(AbstractInitFeaturesEncoder):
    def __init__(self, in_channels, out_channels):
        super(AbstractInitFeaturesEncoder, self).__init__()
        self.linear1 = torch.nn.Linear(in_channels, out_channels)
        self.linear2 = torch.nn.Linear(out_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.BN1 = GraphNorm(out_channels)
        self.BN2 = GraphNorm(out_channels)

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        try:
            x = data.x_0
        except:
            x = data.x
        x = self.relu(self.BN1(self.linear1(x), batch=data.batch))
        x = self.linear2(x)

        data.x = x
        data.x_0 = x
        return data


class BaseEdgeFeatureEncoder(AbstractInitFeaturesEncoder):
    def __init__(self, in_channels, out_channels):
        super(AbstractInitFeaturesEncoder, self).__init__()
        self.linear1 = torch.nn.Linear(in_channels, out_channels)
        self.linear2 = torch.nn.Linear(out_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.BN = GraphNorm(out_channels)

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        x_1 = self.relu(self.linear1(data.x_1))
        x_1 = self.relu(self.linear2(x_1))
        data.x_1 = self.BN(x_1, batch=data.batch_1)
        return data


class BaseFaceFeatureEncoder(AbstractInitFeaturesEncoder):
    def __init__(
        self,
        in_channels_0,
        in_channels_1,
        out_channels_0,
        out_channels_1,
        in_channels_2=None,
        out_channels_2=None,
    ):
        super(AbstractInitFeaturesEncoder, self).__init__()
        self.linear0 = torch.nn.Linear(in_channels_0, out_channels_0)
        self.linear1 = torch.nn.Linear(in_channels_1, out_channels_1)
        if in_channels_2 is not None:
            self.linear2 = torch.nn.Linear(in_channels_2, out_channels_2)

        self.BN0 = GraphNorm(out_channels_0)
        self.BN1 = GraphNorm(out_channels_1)
        if in_channels_2 is not None:
            self.BN2 = GraphNorm(out_channels_2)

        self.relu = torch.nn.ReLU()

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        data.x_0 = self.relu(self.BN0(self.linear0(data.x_0), batch=data.batch))
        data.x_1 = self.relu(self.BN1(self.linear1(data.x_1), batch=data.batch_1))

        if hasattr(data, "x_2") and hasattr(self, "linear2"):
            x_2 = self.linear2(data.x_2)
            if x_2.shape[0] > 0:
                x_2 = self.BN2(x_2, batch=data.batch_2)
            else:
                x_2 = self.BN2(x_2)

            data.x_2 = self.relu(x_2)

        return data


class Node2EdgeProjector(AbstractInitFeaturesEncoder):
    def __init__(
        self,
    ):
        super(AbstractInitFeaturesEncoder, self).__init__()
        # self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        # Project node-level features to edge-level
        data.x_1 = data.x_1 + torch.mm(data.incidence_1.to_dense().T, data.x_0)

        # data.x = self.linear(data.x)
        return data
