import torch
import torch_geometric


class ProjectionLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()

    def lift_features(self, data: torch_geometric.data.Data) -> dict:
        data["x_0"] = data.x

        keys = sorted(
            [
                key.split("_")[1]
                for key in data.keys()
                if ("incidence" in key and "0" not in key)
            ]
        )
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data["x_1"] = data.edge_attr
            keys.remove("1")
        for elem in keys:
            idx_to_project = 0 if elem == "hyperedges" else int(elem) - 1
            data["x_" + elem] = torch.matmul(
                data["incidence_" + elem].t(), data[f"x_{idx_to_project}"]
            )

        return data

    def forward(self, data: torch_geometric.data.Data):
        data = self.lift_features(data)
        return data

    def __call__(self, data):
        return self.forward(data)
