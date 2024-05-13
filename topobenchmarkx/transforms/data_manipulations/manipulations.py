import torch
import torch_geometric
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import one_hot


class IdentityTransform(torch_geometric.transforms.BaseTransform):
    r"""An identity transform that does nothing to the input data."""

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "domain2domain"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The (un)transformed data.
        """
        return data


class InfereKNNConnectivity(torch_geometric.transforms.BaseTransform):
    r"""A transform that generates the k-nearest neighbor connectivity of the
    input point cloud."""

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "infere_knn_connectivity"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.
        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """

        edge_index = knn_graph(data.x, **self.parameters["args"])

        # Remove duplicates
        data.edge_index = edge_index
        return data


class InfereRadiusConnectivity(torch_geometric.transforms.BaseTransform):
    r"""A transform that generates the radius connectivity of the input point
    cloud."""

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "infere_radius_connectivity"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.
        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        data.edge_index = radius_graph(data.x, **self.parameters["args"])
        return data


class EqualGausFeatures(torch_geometric.transforms.BaseTransform):
    r"""A transform that generates equal Gaussian features for all nodes in the
    input graph.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the transform.
    """

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

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        data.x = self.feature_vector.expand(data.num_nodes, -1)
        return data


class NodeFeaturesToFloat(torch_geometric.transforms.BaseTransform):
    r"""A transform that converts the node features of the input graph to float.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "map_node_features_to_float"

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        data.x = data.x.float()
        return data


class NodeDegrees(torch_geometric.transforms.BaseTransform):
    r"""A transform that calculates the node degrees of the input graph.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "node_degrees"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        field_to_process = [
            key
            for key in data
            for field_substring in self.parameters["selected_fields"]
            if field_substring in key and key != "incidence_0"
        ]
        for field in field_to_process:
            data = self.calculate_node_degrees(data, field)

        return data

    def calculate_node_degrees(
        self, data: torch_geometric.data.Data, field: str
    ) -> torch_geometric.data.Data:
        r"""Calculate the node degrees of the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.
        field : str
            The field to calculate the node degrees.

        Returns
        -------
        torch_geometric.data.Data
        """
        if data[field].is_sparse:
            degrees = abs(data[field].to_dense()).sum(1)
        else:
            assert (
                field == "edge_index"
            ), "Following logic of finding degrees is only implemented for edge_index"
            degrees = (
                torch_geometric.utils.to_dense_adj(
                    data[field],
                    max_num_nodes=data["x"].shape[0],  # data["num_nodes"]
                )
                .squeeze(0)
                .sum(1)
            )

        if "incidence" in field:
            field_name = (
                str(int(field.split("_")[1]) - 1) + "_cell" + "_degrees"
            )
        else:
            field_name = "node_degrees"

        data[field_name] = degrees.unsqueeze(1)
        return data


class KeepOnlyConnectedComponent(torch_geometric.transforms.BaseTransform):
    """A transform that keeps only the largest connected components of the
    input graph.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "keep_connected_component"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data):
        """Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        from torch_geometric.transforms import LargestConnectedComponents

        # torch_geometric.transforms.largest_connected_components()
        num_components = self.parameters["num_components"]
        lcc = LargestConnectedComponents(
            num_components=num_components, connection="strong"
        )
        data = lcc(data)
        return data


class CalculateSimplicialCurvature(torch_geometric.transforms.BaseTransform):
    """A transform that calculates the simplicial curvature of the input graph.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "simplicial_curvature"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data):
        """Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        data = self.one_cell_curvature(data)
        data = self.zero_cell_curvature(data)
        data = self.two_cell_curvature(data)
        return data

    def zero_cell_curvature(
        self,
        data: torch_geometric.data.Data,
    ) -> torch_geometric.data.Data:
        """Calculate the zero cell curvature of the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            Data with the zero cell curvature.
        """
        data["0_cell_curvature"] = torch.mm(
            abs(data["incidence_1"]), data["1_cell_curvature"]
        )
        return data

    def one_cell_curvature(
        self,
        data: torch_geometric.data.Data,
    ) -> torch_geometric.data.Data:
        r"""Calculate the one cell curvature of the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            Data with the one cell curvature.
        """
        data["1_cell_curvature"] = (
            4
            - torch.mm(abs(data["incidence_1"]).T, data["0_cell_degrees"])
            + 3 * data["1_cell_degrees"]
        )
        return data

    def two_cell_curvature(
        self,
        data: torch_geometric.data.Data,
    ) -> torch_geometric.data.Data:
        r"""Calculate the two cell curvature of the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            Data with the two cell curvature.
        """
        # Term 1 is simply the degree of the 2-cell (i.e. each triangle belong to n tetrahedrons)
        term1 = data["2_cell_degrees"]
        # Find triangles that belong to multiple tetrahedrons
        two_cell_degrees = data["2_cell_degrees"].clone()
        idx = torch.where(data["2_cell_degrees"] > 1)[0]
        two_cell_degrees[idx] = 0
        up = data["incidence_3"].to_dense() @ data["incidence_3"].to_dense().T
        down = (
            data["incidence_2"].to_dense().T @ data["incidence_2"].to_dense()
        )
        mask = torch.eye(up.size()[0]).bool()
        up.masked_fill_(mask, 0)
        down.masked_fill_(mask, 0)
        diff = (down - up) * 1
        term2 = diff.sum(1, keepdim=True)
        data["2_cell_curvature"] = 3 + term1 - term2
        return data


class OneHotDegreeFeatures(torch_geometric.transforms.BaseTransform):
    r"""A transform that adds the node degree as one hot encodings to the node
    features.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "one_hot_degree_features"
        self.deg_field = kwargs["degrees_fields"]
        self.features_fields = kwargs["features_fields"]
        self.transform = OneHotDegree(max_degree=kwargs["max_degrees"])

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        data = self.transform.forward(
            data,
            degrees_field=self.deg_field,
            features_field=self.features_fields,
        )

        return data


class OneHotDegree(torch_geometric.transforms.BaseTransform):
    r"""Adds the node degree as one hot encodings to the node features.

    Parameters
    ----------
    max_degree : int
        The maximum degree of the graph.
    cat : bool, optional
        If set to `True`, the one hot encodings are concatenated to the node features.
    """

    def __init__(
        self,
        max_degree: int,
        cat: bool = False,
    ) -> None:
        self.max_degree = max_degree
        self.cat = cat

    def forward(
        self,
        data: torch_geometric.data.Data,
        degrees_field: str,
        features_field: str,
    ) -> torch_geometric.data.Data:
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.
        degrees_field : str
            The field containing the node degrees.
        features_field : str
            The field containing the node features.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
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
    r"""A transform that keeps only the selected fields of the input data.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "keep_selected_data_fields"
        self.parameters = kwargs

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        # Keeps all the fields
        fields_to_keep = (
            self.parameters["base_fields"]
            + self.parameters["preserved_fields"]
        )
        # if len(self.parameters["keep_fields"]) == 1:
        #     return data

        # else:
        for key in data:
            if key not in fields_to_keep:
                del data[key]
        return data
