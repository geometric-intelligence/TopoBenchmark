from abc import abstractmethod

import torch
import torch_geometric


class GraphLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self):
        super().__init__()

    def preserve_fields(self, data: torch_geometric.data.Data) -> dict:
        preserved_fields = {}
        for key, value in data.items():
            preserved_fields[key] = value
        return preserved_fields

    def lift_features(self, initial_data: dict, lifted_topology: dict) -> dict:
        features = {}
        features["x_0"] = initial_data["x"]
        keys = sorted(
            [
                key.split("_")[1]
                for key in lifted_topology.keys()
                if ("incidence" in key and "0" not in key)
            ]
        )
        # TODO: revise this to allow using existing edge attributes (specially when having directed edges)
        # if "1" in keys and ("edge_attr" in initial_data or "x_1" in initial_data):
        #     edge_attr = initial_data["edge_attr"] if "edge_attr" in initial_data else initial_data["x_1"]
        #     if edge_attr.shape[0] == lifted_topology["incidence_1"].shape[1]:
        #         features["x_1"] = edge_attr
        #         keys.remove("1")
        for elem in keys:
            idx_to_project = 0 if elem == "hyperedges" else int(elem) - 1
            features["x_" + elem] = torch.matmul(
                lifted_topology["incidence_" + elem].t(),
                features[f"x_{idx_to_project}"],
            )
        return features

    @abstractmethod
    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        raise NotImplementedError

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        initial_data = self.preserve_fields(data)
        lifted_topology = self.lift_topology(data)
        lifted_features = self.lift_features(initial_data, lifted_topology)
        lifted_data = torch_geometric.data.Data(
            **initial_data, **lifted_topology, **lifted_features
        )
        return lifted_data
