from abc import abstractmethod

import torch_geometric


class Graph2Domain(torch_geometric.transforms.BaseTransform):
    def __init__(self):
        super().__init__()

    def preserve_fields(self, data: torch_geometric.data.Data) -> dict:
        preserved_fields = {}
        for key, value in data.items():
            preserved_fields[key] = value
        return preserved_fields

    # @abstractmethod
    # def lift_features(self, data: torch_geometric.data.Data, lifted_topology) -> dict:
    #     raise NotImplementedError

    @abstractmethod
    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        raise NotImplementedError

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        initial_data = self.preserve_fields(data)
        lifted_topology = self.lift_topology(data)
        # lifted_features = self.lift_features(data, lifted_topology)
        lifted_data = torch_geometric.data.Data(
            **initial_data,
            **lifted_topology,  # **lifted_features
        )
        return lifted_data
