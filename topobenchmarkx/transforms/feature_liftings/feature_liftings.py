import torch
import torch_geometric


class BaseTopologyLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()

    def lift_features(self, data: torch_geometric.data.Data) -> dict:
        data["x_0"] = data.x

        # TODO: Check if that is correct
        keys = [key for key in data.keys() if "incidence" in key]
        for i, _ in enumerate(keys):
            data[f"x_{i + 1}"] = torch.matmul(
                data[f"incidence_{i + 1}"].t(), data[f"x_{i}"]
            )

        return data

    def forward(self, data: torch_geometric.data.Data):
        data = self.lift_features(data)
        return data

    def __call__(self, data):
        return self.forward(data)
