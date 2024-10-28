"""Attention to graph transform."""

import torch
import torch_geometric


class Attention2Graph(torch_geometric.transforms.BaseTransform):
    r"""A transform that sparcifies the attention.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "attention2graph"
        self.parameters = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, parameters={self.parameters!r})"

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

        # Reshape the initial attention scores to the original shape
        attention_shape = data.attention_shape
        attention_scores = data.attention_scores.reshape(attention_shape)
        
        mask = attention_scores > self.parameters["threshold"]

        edge_index = torch.stack(torch.where(mask==1))

        edge_index = torch_geometric.utils.remove_self_loops(edge_index)[0]

        edge_index = torch_geometric.utils.to_undirected(edge_index)

        data.edge_index = edge_index
       

        return data

   