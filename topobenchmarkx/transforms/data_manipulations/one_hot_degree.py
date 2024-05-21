import torch
import torch_geometric
from torch_geometric.utils import one_hot

class OneHotDegree(torch_geometric.transforms.BaseTransform):
    r"""Adds the node degree as one hot encodings to the node features.

    Args:
        max_degree (int): The maximum degree of the graph.
        cat (bool, optional): If set to `True`, the one hot encodings are concatenated to the node features. (default: False)
    """
    def __init__(
        self,
        max_degree: int,
        cat: bool = False,
        **kwargs,
    ) -> None:
        self.max_degree = max_degree
        self.cat = cat
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_degree={self.max_degree}, cat={self.cat})"

    def forward(
        self,
        data: torch_geometric.data.Data,
        degrees_field: str,
        features_field: str,
    ) -> torch_geometric.data.Data:
        r"""Apply the transform to the input data.

        Args:
            data (torch_geometric.data.Data): The input data.
            degrees_field (str): The field containing the node degrees.
            features_field (str): The field containing the node features.
        Returns:
            torch_geometric.data.Data: The transformed data.
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