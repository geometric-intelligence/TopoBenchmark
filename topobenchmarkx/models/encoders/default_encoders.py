import torch
import torch_geometric

from topobenchmarkx.models.abstractions.encoder import AbstractInitFeaturesEncoder


class BaseNodeFeatureEncoder(AbstractInitFeaturesEncoder):
    def __init__(self, in_channels, out_channels):
        super(AbstractInitFeaturesEncoder, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        data.x = self.linear(data.x)
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
