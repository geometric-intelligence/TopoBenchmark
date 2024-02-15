import torch
import torch_geometric


class IdentityTransform(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.type = "domain2domain"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data) -> dict:
        return data

    def __call__(self, data):
        return self.forward(data)


class DataFieldsToDense(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.type = "datafields2dense"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data) -> dict:
        # Move the data fields to dense if data[key] is sparse
        for key, value in data.items():
            if hasattr(value, "to_dense"):
                data[key] = value.to_dense()

        # # Workaround for the batch attribute
        # for key, value in data.items():
        #     # Check if values is a tensor
        #     if type(data[key]) == torch.Tensor:
        #         data[key] = value.unsqueeze(1)
        #     else:
        #         data[key] = torch.tensor(data[key]).unsqueeze(0)

        return data

    def __call__(self, data):
        return self.forward(data)


class GraphNodeDegrees(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.type = "graph_node_degrees"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data) -> dict:
        degrees = torch_geometric.utils.degree(data.edge_index[0])
        assert degrees.size(0) == data.x.size(
            0
        ), "Node degrees and node features must have the same size!"
        data["node_degrees"] = degrees
        return data

    def __call__(self, data):
        return self.forward(data)
