import torch_geometric

from topobenchmarkx.transforms import TRANSFORMS


class DataTransform(torch_geometric.transforms.BaseTransform):
    r"""Abstract class that provides an interface to define a custom data
    lifting.

    Args:
        transform_name (str): The name of the transform to be used.
        **kwargs: Additional arguments for the class.
    """

    def __init__(self, transform_name, **kwargs):
        super().__init__()

        kwargs["transform_name"] = transform_name
        self.parameters = kwargs

        self.transform = (
            TRANSFORMS[transform_name](**kwargs)
            if transform_name is not None
            else None
        )

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Forward pass of the lifting.

        Args:
            data (torch_geometric.data.Data): The input data to be lifted.
        Returns:
            transformed_data (torch_geometric.data.Data): The lifted data.
        """
        transformed_data = self.transform(data)
        return transformed_data
