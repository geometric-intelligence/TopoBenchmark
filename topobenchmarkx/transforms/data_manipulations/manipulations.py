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


class NodeFeaturesToFloat(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.type = "map_node_features_to_float"

    def forward(self, data: torch_geometric.data.Data) -> dict:
        data.x = data.x.float()
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
        field_to_process = []
        for key in data.keys():
            for field_substring in self.parameters["selected_fields"]:
                if field_substring in key and key != "incidence_0":
                    field_to_process.append(key)

        for field in field_to_process:
            data = self.calculate_node_degrees(data, field)

        return data

    def calculate_node_degrees(
        self, data: torch_geometric.data.Data, field: str
    ) -> torch.Tensor:
        if data[field].is_sparse:
            degrees = abs(data[field].to_dense()).sum(1)
        else:
            assert (
                field == "edge_index"
            ), "Following logic of finding degrees is only implemented for edge_index"
            degrees = (
                torch_geometric.utils.to_dense_adj(
                    data[field], max_num_nodes=data["x"].shape[0]  # data["num_nodes"]
                )
                .squeeze(0)
                .sum(1)
            )

        if "incidence" in field:
            field_name = str(int(field.split("_")[1]) - 1) + "_cell" + "_degrees"
        else:
            field_name = "node_degrees"

        data[field_name] = degrees.unsqueeze(1)
        return data

    def __call__(self, data):
        return self.forward(data)


class KeepOnlyConnectedComponent(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.type = "keep_connected_component"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data) -> dict:
        from torch_geometric.transforms import LargestConnectedComponents

        # torch_geometric.transforms.largest_connected_components()
        num_components = self.parameters["num_components"]
        lcc = LargestConnectedComponents(
            num_components=num_components, connection="strong"
        )
        data = lcc(data)
        return data

    def __call__(self, data):
        return self.forward(data)


class CalculateSimplicialCurvature(torch_geometric.transforms.BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.type = "simplicial_curvature"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data) -> dict:
        data = self.one_cell_curvature(data)
        data = self.zero_cell_curvature(data)
        data = self.two_cell_curvature(data)
        return data

    def zero_cell_curvature(
        self,
        data: torch_geometric.data.Data,
    ) -> torch_geometric.data.Data:
        data["0_cell_curvature"] = torch.mm(
            abs(data["incidence_1"]), data["1_cell_curvature"]
        )
        return data

    def one_cell_curvature(
        self,
        data: torch_geometric.data.Data,
    ) -> torch_geometric.data.Data:
        data["1_cell_curvature"] = (
            4
            - torch.mm(abs(data["incidence_1"]).T, data["0_cell_degrees"])
            + 3 * data["1_cell_degrees"]
        )
        return data

    def two_cell_curvature(
        self,
        data: torch_geometric.data.Data,
    ) -> torch.Tensor:
        # Term 1 is simply the degree of the 2-cell (i.e. each triangle belong to n tetrahedrons)
        term1 = data["2_cell_degrees"]

        # set diag values to zero

        # Find triangles that belong to multiple tetrahedrons
        two_cell_degrees = data["2_cell_degrees"].clone()
        idx = torch.where(data["2_cell_degrees"] > 1)[0]
        two_cell_degrees[idx] = 0

        up = data["incidence_3"].to_dense() @ data["incidence_3"].to_dense().T
        down = data["incidence_2"].to_dense().T @ data["incidence_2"].to_dense()
        mask = torch.eye(up.size()[0]).bool()
        up.masked_fill_(mask, 0)
        down.masked_fill_(mask, 0)
        diff = (down - up) * 1
        term2 = diff.sum(1, keepdim=True)

        # # Find all triangles that belong to at least one tetrahedron
        # idx = torch.where(two_cell_degrees > 0)[0] #torch.where(data["2_cell_degrees"] > 0)[0]

        # # Edges to Triangles incidence matrix
        # incidence_2 = data["incidence_2"].to_dense().clone()

        # # Keep only those triangles that belong to at least one tetrahedron
        # incidence_2_subset = incidence_2[:, idx]

        # # Find 1-cell (edge) degrees aka find the number of triangles that every 1-cell belongs to
        # one_cell_degreees_subset = incidence_2_subset.sum(1, keepdim=True)

        # # Check the condition
        # one_cell_degrees_conditioned = (
        #     one_cell_degreees_subset == data["1_cell_degrees"]
        # ) * one_cell_degreees_subset
        # term2 = torch.mm(abs(data["incidence_2"]).T, one_cell_degrees_conditioned)
        data["2_cell_curvature"] = 3 + term1 - term2

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

        if len(deg.shape) == 2:
            deg = deg.squeeze(1)

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
        # Keeps all the fields
        if len(self.parameters["keep_fields"]) == 1:
            return data

        else:
            for key, _ in data.items():
                if key not in self.parameters["keep_fields"]:
                    del data[key]
        return data

    def __call__(self, data):
        return self.forward(data)
