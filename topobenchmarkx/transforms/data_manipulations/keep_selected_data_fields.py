import torch_geometric


class KeepSelectedDataFields(torch_geometric.transforms.BaseTransform):
    r"""A transform that keeps only the selected fields of the input data.

    Args:
        kwargs (optional): Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "keep_selected_data_fields"
        self.parameters = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, parameters={self.parameters!r})"

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Args:
            data (torch_geometric.data.Data): The input data.
        Returns:
            torch_geometric.data.Data: The transformed data.
        """
        # Keeps all the fields
        fields_to_keep = (
            self.parameters["base_fields"]
            + self.parameters["preserved_fields"]
        )

        for key in data.keys():
            if key not in fields_to_keep:
                del data[key]
        return data
