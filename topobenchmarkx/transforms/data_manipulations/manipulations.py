import torch
import torch_geometric
from torch_geometric.utils import one_hot


class IdentityTransform(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.type = "domain2domain"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data) -> dict:
        return data

    def __call__(self, data):
        return self.forward(data)


class EqualGausFeatures(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.type = "generate_non_informative_features"

        # Torch generate feature vector from gaus distribution
        self.mean = kwargs["mean"]
        self.std = kwargs["std"]
        self.feature_vector = kwargs["num_features"]
        self.feature_vector = torch.normal(
            mean=self.mean, std=self.std, size=(1, self.feature_vector)
        )

    def forward(self, data: torch_geometric.data.Data) -> dict:
        data.x = self.feature_vector.expand(data.num_nodes, -1)
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


class OneHotDegreeFeatures(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.type = "one_hot_degree_features"
        self.deg_field = kwargs["degrees_fields"]
        self.features_fields = kwargs["features_fields"]
        self.transform = OneHotDegree(max_degree=kwargs["max_degrees"])

    def forward(self, data: torch_geometric.data.Data) -> dict:
        data = self.transform.forward(
            data, degrees_field=self.deg_field, features_field=self.features_fields
        )

        return data

    def __call__(self, data):
        return self.forward(data)


class OneHotDegree(torch_geometric.transforms.BaseTransform):
    r"""Adds the node degree as one hot encodings to the node features
    (functional name: :obj:`one_hot_degree`).

    Args:
        max_degree (int): Maximum degree.
        in_degree (bool, optional): If set to :obj:`True`, will compute the
            in-degree of nodes instead of the out-degree.
            (default: :obj:`False`)
        cat (bool, optional): Concat node degrees to node features instead
            of replacing them. (default: :obj:`True`)
    """

    def __init__(
        self,
        max_degree: int,
        cat: bool = False,
    ) -> None:
        self.max_degree = max_degree
        self.cat = cat

    def forward(
        self, data: torch_geometric.data.Data, degrees_field: str, features_field: str
    ) -> torch_geometric.data.Data:
        assert data.edge_index is not None

        deg = data[degrees_field].to(torch.long)
        deg = one_hot(deg, num_classes=self.max_degree + 1)

        if self.cat:
            x = data[features_field]
            x = x.view(-1, 1) if x.dim() == 1 else x
            data[features_field] = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data[features_field] = deg

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.max_degree})"


class KeepSelectedDataFields(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.type = "keep_selected_data_fields"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data) -> dict:
        self.parameters["keep_fields"]
        for key, _ in data.items():
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
        fields = [key for key in data.keys() if "x" in key and len(key) == 1]

        for field in fields:
            assert (
                self.parameters["expected_number_of_features"]
                == data[field][:, dim_slice:].shape[1]
            ), "The expected number of features does not match the number of features in the data"
            data[field] = data[field][:, dim_slice:]

        return data

    def __call__(self, data):
        return self.forward(data)
