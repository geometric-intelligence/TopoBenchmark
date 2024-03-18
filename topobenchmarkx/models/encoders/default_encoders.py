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
        x = self.dropout(self.linear1(x))
        x = self.BN(x, batch=batch) if batch.shape[0] > 0 else self.BN(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class BaseFeatureEncoder(AbstractInitFeaturesEncoder):
    def __init__(self, in_channels, out_channels, proj_dropout=0, selected_dimensions=None):
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
                BaseEncoder(self.in_channels[i], self.out_channels, dropout=proj_dropout),
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
    def __init__(self, in_channels, out_channels, proj_dropout=0, selected_dimensions=None):
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
                    BaseEncoder(self.in_channels[i], self.out_channels, dropout=proj_dropout),
                )
            else:
                setattr(
                    self,
                    f"encoder_{i}",
                    Perceiver(dim=self.out_channels, depth=idx+1, cross_heads=4, 
                              cross_dim_head=self.out_channels, latent_dim_head=self.out_channels),
                )

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        
        if not hasattr(data, "x_0"):
            data.x_0 = data.x
        
        for idx, i in enumerate(self.dimensions):
            if idx == 0:
                if hasattr(data, f"x_{i}") and hasattr(self, f"encoder_{i}"):
                    batch = data.batch if i == 0 else getattr(data, f"batch_{i}")
                    data[f"x_{i}"] = getattr(self, f"encoder_{i}")(data[f"x_{i}"], batch)
            else:
                if hasattr(data, f"x_{i}") and hasattr(self, f"encoder_{i}"): 
                    cell_features = data["x_0"][data[f"x_{i}"].long()]
                    data[f"x_{i}"] = getattr(self, f"encoder_{i}")(cell_features)
                else: 
                    data[f"x_{i}"] = torch.tensor([], device=data.x_0.device)
        return data


# class BaseShallowEncoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, hidden_layers=1):
#         super().__init__()
#         self.linear1 = torch.nn.Linear(in_channels, out_channels)        
#         self.relu = torch.nn.ELU()
        

#     def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
#         x = self.linear1(x)
#         x = self.relu(x)
#         return x
    
# class BaseShallowFeatureEncoder(AbstractInitFeaturesEncoder):
#     def __init__(self, in_channels, out_channels, selected_dimensions=None):
#         super(AbstractInitFeaturesEncoder, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.dimensions = (
#             selected_dimensions
#             if selected_dimensions is not None
#             else range(len(self.in_channels))
#         )
#         for i in self.dimensions:
#             setattr(
#                 self,
#                 f"encoder_{i}",
#                 BaseShallowEncoder(self.in_channels[i], self.out_channels),
#             )

#     def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        
#         if not hasattr(data, "x_0"):
#             data.x_0 = data.x
        
#         for i in self.dimensions:
#             if hasattr(data, f"x_{i}") and hasattr(self, f"encoder_{i}"):
#                 batch = data.batch if i == 0 else getattr(data, f"batch_{i}")
#                 data[f"x_{i}"] = getattr(self, f"encoder_{i}")(data[f"x_{i}"], batch)
#         return data


# class BaseNodeFeatureEncoder(AbstractInitFeaturesEncoder):
#     def __init__(self, in_channels, out_channels):
#         super(AbstractInitFeaturesEncoder, self).__init__()
#         self.linear1 = torch.nn.Linear(in_channels, out_channels)
#         self.linear2 = torch.nn.Linear(out_channels, out_channels)
#         self.relu = torch.nn.ReLU()
#         self.BN = GraphNorm(out_channels)

#     def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
#         x = data.x_0 if hasattr(data, "x_0") else data.x
#         x = self.BN(self.linear1(x), batch=data.batch)
#         x = self.linear2(x)

#         data.x = x
#         data.x_0 = x
#         return data


# class BaseEdgeFeatureEncoder(AbstractInitFeaturesEncoder):
#     def __init__(self, in_channels, out_channels):
#         super(AbstractInitFeaturesEncoder, self).__init__()
#         self.linear1 = torch.nn.Linear(in_channels, out_channels)
#         self.linear2 = torch.nn.Linear(out_channels, out_channels)
#         self.relu = torch.nn.ReLU()
#         self.BN = GraphNorm(out_channels)

#     def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
#         x_1 = self.relu(self.linear1(data.x_1))
#         x_1 = self.relu(self.linear2(x_1))
#         data.x_1 = self.BN(x_1, batch=data.batch_1)
#         return data


# class BaseFaceFeatureEncoder(AbstractInitFeaturesEncoder):
#     def __init__(
#         self,
#         in_channels_0,
#         in_channels_1,
#         out_channels_0,
#         out_channels_1,
#         in_channels_2=None,
#         out_channels_2=None,
#     ):
#         super(AbstractInitFeaturesEncoder, self).__init__()
#         self.linear0 = torch.nn.Linear(in_channels_0, out_channels_0)
#         self.linear1 = torch.nn.Linear(in_channels_1, out_channels_1)
#         if in_channels_2 is not None:
#             self.linear2 = torch.nn.Linear(in_channels_2, out_channels_2)

#         self.BN0 = GraphNorm(out_channels_0)
#         self.BN1 = GraphNorm(out_channels_1)
#         if in_channels_2 is not None:
#             self.BN2 = GraphNorm(out_channels_2)

#         self.relu = torch.nn.ReLU()

#     def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
#         data.x_0 = self.relu(self.BN0(self.linear0(data.x_0), batch=data.batch))
#         data.x_1 = self.relu(self.BN1(self.linear1(data.x_1), batch=data.batch_1))

#         if hasattr(data, "x_2") and hasattr(self, "linear2"):
#             x_2 = self.linear2(data.x_2)
#             if x_2.shape[0] > 0:
#                 x_2 = self.BN2(x_2, batch=data.batch_2)
#             else:
#                 x_2 = self.BN2(x_2)

#             data.x_2 = self.relu(x_2)

#         return data


# class Node2EdgeProjector(AbstractInitFeaturesEncoder):
#     def __init__(
#         self,
#     ):
#         super(AbstractInitFeaturesEncoder, self).__init__()
#         # self.linear = torch.nn.Linear(in_channels, out_channels)

#     def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
#         # Project node-level features to edge-level
#         data.x_1 = data.x_1 + torch.mm(data.incidence_1.to_dense().T, data.x_0)

#         # data.x = self.linear(data.x)
#         return data
