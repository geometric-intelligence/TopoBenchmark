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
        raise NotImplementedError
        # # Move the data fields to dense if data[key] is sparse
        # for key, value in data.items():
        #     if hasattr(value, "to_dense"):
        #         data[key] = value.to_dense()

        # # # Workaround for the batch attribute
        # # for key, value in data.items():
        # #     # Check if values is a tensor
        # #     if type(data[key]) == torch.Tensor:
        # #         data[key] = value.unsqueeze(1)
        # #     else:
        # #         data[key] = torch.tensor(data[key]).unsqueeze(0)

        # return data

    def __call__(self, data):
        return self.forward(data)


class NodeDegrees(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.type = "node_degrees"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data) -> dict:
        for field in self.parameters["selected_fields"]:
            data = self.calculate_node_degrees(data, field)

        return data

    def calculate_node_degrees(
        self, data: torch_geometric.data.Data, field: str
    ) -> torch.Tensor:
        if data[field].is_sparse:
            indices = data[field].indices()
            degrees = torch_geometric.utils.degree(indices[0])
        else:
            degrees = torch_geometric.utils.degree(data[field][0])

        # assert degrees.size(0) == data.x.size(
        #     0
        # ), "Node degrees and node features must have the same size!"
        if field == "edge_index":
            name_degrees = "node_degrees"
        elif field == "incidence_1":
            name_degrees = "x_0_degrees"
        elif field == "incidence_2":
            name_degrees = "x_1_degrees"
        else:
            pass

        data[name_degrees] = degrees
        return data

    def __call__(self, data):
        return self.forward(data)


class KeepSelectedDataFields(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.type = "keep_selected_data_fields"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data) -> dict:
        self.parameters["keep_fields"]
        for key, value in data.items():
            if key not in self.parameters["keep_fields"]:
                del data[key]
        return data

    def __call__(self, data):
        return self.forward(data)


class RemoveExtraFeatureFromProteins(torch_geometric.transforms.BaseTransform):
    """Remove extra features from the proteins dataset

    While loading with pretransform, the proteins dataset has extra features
    that are not present in the original dataset. This transform removes the
    extra features.
    The extra features is supposed to be added when use_node_attr=True, but
    even with use_node_attr=False and passed pre_transforms, the extra features
    are removed from data.x after pre_transforms are applied.

    HardCoded solution
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "remove_extra_features_from_proteins"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data) -> dict:
        dim_slice = self.parameters["remove_first_n_features"]
        search_field = self.parameters["search_field"]
        # Find all the fields that contain the search_field
        fields = [key for key in data.keys() if search_field in key and len(key) == 3]

        for field in fields:
            data[field] = data[field][:, dim_slice:]
        return data

    def __call__(self, data):
        return self.forward(data)
