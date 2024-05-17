import torch_geometric
from topobenchmarkx.transforms.data_manipulations.one_hot_degree import OneHotDegree



class OneHotDegreeFeatures(torch_geometric.transforms.BaseTransform):
    r"""A transform that adds the node degree as one hot encodings to the node
    features.

    Args:
        kwargs (optional): Parameters for the base transform.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.type = "one_hot_degree_features"
        self.deg_field = kwargs["degrees_fields"]
        self.features_fields = kwargs["features_fields"]
        self.transform = OneHotDegree(max_degree=kwargs["max_degrees"])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, degrees_field={self.deg_field!r}, features_field={self.features_fields!r})"
    
    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Args:
            data (torch_geometric.data.Data): The input data.
        Returns:
            torch_geometric.data.Data: The transformed data.
        """
        data = self.transform.forward(
            data,
            degrees_field=self.deg_field,
            features_field=self.features_fields,
        )

        return data