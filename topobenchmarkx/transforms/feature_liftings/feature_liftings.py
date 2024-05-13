import torch
import torch_geometric


class ProjectionSum(torch_geometric.transforms.BaseTransform):
    r"""Lifts r-cell features to r+1-cells by projection.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def lift_features(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Projects r-cell features of a graph to r+1-cell structures using the
        incidence matrix.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data.
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

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data.
        """
        data = self.lift_features(data)
        return data


class ConcatentionLifting(torch_geometric.transforms.BaseTransform):
    r"""Lifts r-cell features to r+1-cells by concatenation.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def lift_features(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Concatenates r-cell features to r+1-cell structures using the
        incidence matrix.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data.
        """

        keys = sorted(
            [key.split("_")[1] for key in data if "incidence" in key]
        )
        for elem in keys:
            if f"x_{elem}" not in data:
                idx_to_project = 0 if elem == "hyperedges" else int(elem) - 1
                incidence = data["incidence_" + elem]
                _, n = incidence.shape

                if n != 0:
                    idxs = []
                    for n_feature in range(n):
                        idxs_for_feature = incidence.indices()[
                            0, incidence.indices()[1, :] == n_feature
                        ]
                        idxs.append(torch.sort(idxs_for_feature)[0])

                    idxs = torch.stack(idxs, dim=0)
                    values = data[f"x_{idx_to_project}"][idxs].view(n, -1)
                else:
                    values = torch.tensor([])

                data["x_" + elem] = values
        return data

    def forward(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Applies the lifting to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data.
        """
        data = self.lift_features(data)
        return data


class SetLifting(torch_geometric.transforms.BaseTransform):
    r"""Lifts r-cell features to r+1-cells by set operations.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def lift_features(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Concatenates r-cell features to r+1-cell structures using the
        incidence matrix.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data.
        """

        keys = sorted(
            [key.split("_")[1] for key in data if "incidence" in key]
        )
        for elem in keys:
            if f"x_{elem}" not in data:
                # idx_to_project = 0 if elem == "hyperedges" else int(elem) - 1
                incidence = data["incidence_" + elem]
                _, n = incidence.shape

                if n != 0:
                    idxs = []
                    for n_feature in range(n):
                        idxs_for_feature = incidence.indices()[
                            0, incidence.indices()[1, :] == n_feature
                        ]
                        idxs.append(torch.sort(idxs_for_feature)[0])

                    idxs = torch.stack(idxs, dim=0)
                    if elem == "1":
                        values = idxs
                    else:
                        values = torch.sort(
                            torch.unique(
                                data["x_" + str(int(elem) - 1)][idxs].view(
                                    idxs.shape[0], -1
                                ),
                                dim=1,
                            ),
                            dim=1,
                        )[0]
                else:
                    values = torch.tensor([])

                data["x_" + elem] = values
        return data

    def forward(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Applies the lifting to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data.
        """
        data = self.lift_features(data)
        return data
