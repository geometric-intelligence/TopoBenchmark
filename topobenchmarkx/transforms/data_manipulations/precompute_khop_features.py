"""Precompute the features based on KHop neighbourhoods transform."""

import torch
import torch_geometric


class PrecomputeKHopFeatures(torch_geometric.transforms.BaseTransform):
    r"""Class for precomputing the features of a k-hop neighbourhood features transform.

    A transform that computes an aggregation of injective transformations of the k-hop neighbourhood.

    Parameters
    ----------
    max_hop : int
        The maximum hop neighbourhood.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(
        self,
        max_hop: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.type = "precompute_khop_features"
        self.max_hop = max_hop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, max_hop={self.max_hop})"

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
        T = self.max_hop
        K = 2  # TODO Check where to get this degree from
        B = [data[f"incidence_{i+1}"] for i in range(K + 1)]
        UP = [torch.mm(B[i], B[i].T) for i in range(K + 1)]
        DOWN = [torch.mm(B[i].T, B[i]) for i in range(K + 1)]
        Bs = [B[i].to_dense() for i in range(K + 1)]
        Bc = [B[i].T.to_dense() for i in range(K + 1)]

        Bs_new = [torch.ones_like(Bs[0])] * len(Bs)
        Bc_new = [torch.ones_like(Bc[0])] * len(Bc)

        # TODO Normalize the features

        N0 = (UP[0].size())[0]  # (number of 0-simplices)
        N1 = (UP[1].size())[0]  # Number of 1-simplices
        N2 = (DOWN[1].size())[0]  # Number of 2-simplices
        N3 = (DOWN[2].size())[0]  # Number of 3-simplices

        x_is = [
            torch.ones((N0, 1)),
            torch.ones((N1, 1)),
            torch.ones((N2, 1)),
            torch.ones((N3, 1)),
        ]

        # Create a dictionary that stores the i-simplices, the
        # j-th hop features matrix
        x_all = {
            f"x{i}_{j}": torch.ones(1)
            for i in range(K + 1)
            for j in range(T + 1)
        }

        # Compute normalization matrices D_k
        for i in range(K + 1):
            LS = torch.mm(torch.abs(UP[i]), x_is[i]).flatten()
            D_i = LS - torch.diag(torch.abs(UP[i].to_dense())).flatten()
            D_i = torch.diagflat(torch.div(1, torch.sqrt(D_i)))
            # TODO Check this is correct

            if i > 0:
                DOWN[i - 1] = torch.mm(torch.mm(D_i, DOWN[i - 1]), D_i)
                D_i_i_minus_1 = torch.mm(
                    torch.abs(Bc[i - 1]), x_is[i - 1]
                ).flatten()
                D_i_i_minus_1 = torch.diagflat(
                    torch.div(1, torch.sqrt(D_i_i_minus_1))
                )

                D_i_minus_1_i = torch.mm(Bs[i - 1], x_is[i]).flatten()
                D_i_minus_1_i = torch.diagflat(
                    torch.div(1, torch.sqrt(D_i_minus_1_i))
                )

                # D_{i, i-1} B_{i-1} D_{i-1, i}
                Bs_new[i - 1] = torch.mm(
                    torch.mm(D_i_i_minus_1, Bc[i - 1]), D_i_minus_1_i
                )

            if i < K:
                D_i_i_plus_1 = torch.mm(Bs[i], x_is[i + 1]).flatten()
                D_i_i_plus_1 = torch.diagflat(
                    torch.div(1, torch.sqrt(D_i_i_plus_1))
                )

                D_i_plus_1_i = torch.mm(torch.abs(Bc[i]), x_is[i]).flatten()
                D_i_plus_1_i = torch.diagflat(
                    torch.div(1, torch.sqrt(D_i_plus_1_i))
                )

                # D_{i, i-1} B_{i-1} D_{i-1, i}
                Bc_new[i] = torch.mm(
                    torch.mm(D_i_i_plus_1, Bs[i]), D_i_plus_1_i
                )
            # Update UP at the end
            UP[i] = torch.mm(torch.mm(D_i, UP[i]), D_i)

        # Set the information for the 0-hop embeddings
        for i in range(K + 1):
            x_all[f"x{i}_0"] = data[f"x_{i}"]

        # For each hop t=1,...,T
        for t in range(1, T + 1):
            # For each k-simplex, k=0,...,K
            for k in range(K + 1):
                # Set everything to `None` as
                # some representations are not available
                Y_U_1 = None
                Y_L_1 = None
                Y_B_1 = None
                Y_C_1 = None

                adjacencies_embedding = []
                # The highest order simplex does not have a higher order adjacency
                Y_U_1 = torch.mm(UP[k], x_all[f"x{k}_{t-1}"])
                adjacencies_embedding.append(Y_U_1)
                if k < K:
                    Y_C_1 = torch.mm(Bc_new[k], x_all[f"x{k+1}_{t-1}"])

                    adjacencies_embedding.append(Y_C_1)
                # The 0-simplex does not have a lower-adjacency
                if k > 0:
                    Y_L_1 = torch.mm(DOWN[k - 1], x_all[f"x{k}_{t-1}"])
                    Y_B_1 = torch.mm(Bs_new[k - 1], x_all[f"x{k-1}_{t-1}"])
                    adjacencies_embedding.append(Y_L_1)
                    adjacencies_embedding.append(Y_B_1)

                # Concat the feature dimension of the adjacy normalized features
                X = torch.cat(adjacencies_embedding, dim=1)
                x_all[f"x{k}_{t}"] = X
        return torch_geometric.data.Data(**data, **x_all)
