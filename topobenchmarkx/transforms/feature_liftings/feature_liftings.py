import torch
import torch_geometric


class ProjectionLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()

    def lift_features(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        keys = sorted([key.split("_")[1] for key in data.keys() if "incidence" in key])
        for elem in keys:
            if f"x_{elem}" not in data:
                idx_to_project = 0 if elem == "hyperedges" else int(elem) - 1
                data["x_" + elem] = torch.matmul(
                    data["incidence_" + elem].t(),
                    data[f"x_{idx_to_project}"],
                )
        return data

    def forward(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        data = self.lift_features(data)
        return data

    def __call__(self, data):
        return self.forward(data)
