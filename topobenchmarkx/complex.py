import torch


class PlainComplex:
    def __init__(
        self,
        incidence,
        down_laplacian,
        up_laplacian,
        adjacency,
        coadjacency,
        hodge_laplacian,
        features=None,
    ):
        # TODO: allow None with nice error message if callable?

        # TODO: make this private? do not allow for changes in these values?
        self.incidence = incidence
        self.down_laplacian = down_laplacian
        self.up_laplacian = up_laplacian
        self.adjacency = adjacency
        self.coadjacency = coadjacency
        self.hodge_laplacian = hodge_laplacian

        if features is None:
            features = [None for _ in range(len(self.incidence))]
        else:
            for rank, dim in enumerate(self.shape):
                # TODO: make error message more informative
                if (
                    features[rank] is not None
                    and features[rank].shape[0] != dim
                ):
                    raise ValueError("Features have wrong shape.")

        self.features = features

    @property
    def shape(self):
        """Shape of the complex.

        Returns
        -------
        list[int]
        """
        return [incidence.shape[-1] for incidence in self.incidence]

    @property
    def max_rank(self):
        """Maximum rank of the complex.

        Returns
        -------
        int
        """
        return len(self.incidence)

    def update_features(self, rank, values):
        """Update features.

        Parameters
        ----------
        rank : int
            Rank of simplices the features belong to.
        values : array-like
            New features for the rank-simplices.
        """
        self.features[rank] = values

    def reset_features(self):
        """Reset features."""
        self.features = [None for _ in self.features]

    def propagate_values(self, rank, values):
        """Propagate features from a rank to an upper one.

        Parameters
        ----------
        rank : int
            Rank of the simplices the values belong to.
        values : array-like
            Features for the rank-simplices.
        """
        # TODO: can be made much better
        return torch.matmul(torch.abs(self.incidence[rank + 1].t()), values)
