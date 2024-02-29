import torch
import torch_geometric
from torch.nn import BatchNorm1d as BN

from topobenchmarkx.models.abstractions.encoder import AbstractInitFeaturesEncoder


class BaseNodeFeatureEncoder(AbstractInitFeaturesEncoder):
    def __init__(self, in_channels, out_channels):
        super(AbstractInitFeaturesEncoder, self).__init__()
        self.linear1 = torch.nn.Linear(in_channels, out_channels)
        self.linear2 = torch.nn.Linear(out_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.BN1 = BN(out_channels)
        self.BN2 = BN(out_channels)

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        try:
            x = data.x_0
        except:
            x = data.x
        x = self.relu(self.BN1(self.linear1(x)))
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
        self.BN = BN(out_channels)

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        x_1 = self.relu(self.linear1(data.x_1))
        x_1 = self.relu(self.linear2(x_1))
        data.x_1 = self.BN(x_1)
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
