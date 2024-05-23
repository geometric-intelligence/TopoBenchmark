import torch
import torch_geometric


class ProjectionSum(torch_geometric.transforms.BaseTransform):
    r"""Lifts r-cell features to r+1-cells by projection.

    Args:
        kwargs (optional): Additional arguments for the class.
    """
    def __init__(self, **kwargs):
        super().__init__()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def lift_features(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Projects r-cell features of a graph to r+1-cell structures using the
        incidence matrix.

        Args:
            data (torch_geometric.data.Data | dict): The input data to be lifted.
        Returns:
            torch_geometric.data.Data | dict: The data with the lifted features.
        """
        keys = sorted(
            [key.split("_")[1] for key in data if "incidence" in key]
        )
        for elem in keys:
            if f"x_{elem}" not in data:
                idx_to_project = 0 if elem == "hyperedges" else int(elem) - 1
                data["x_" + elem] = torch.matmul(
                    abs(data["incidence_" + elem].t()),
                    data[f"x_{idx_to_project}"],
                )
        return data

    def forward(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Applies the lifting to the input data.

        Args:
            data (torch_geometric.data.Data | dict): The input data to be lifted.
        Returns:
            torch_geometric.data.Data | dict: The lifted data.
        """
        data = self.lift_features(data)
        return data
