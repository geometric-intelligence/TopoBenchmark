import abc


class Data(abc.ABC):
    def __init__(self, incidence, features):
        self.incidence = incidence
        self.features = features

    @abc.abstractmethod
    def keys(self):
        pass

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

    @property
    def shape(self):
        """Shape of the complex.

        Returns
        -------
        list[int]
        """
        return [
            None
            if self.incidence[key] is None
            else self.incidence[key].shape[-1]
            for key in self.keys()
        ]


class ComplexData(Data):
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
        self.down_laplacian = down_laplacian
        self.up_laplacian = up_laplacian
        self.adjacency = adjacency
        self.coadjacency = coadjacency
        self.hodge_laplacian = hodge_laplacian

        if features is None:
            features = [None for _ in range(len(incidence))]
        else:
            for rank, incidence_ in enumerate(incidence):
                # TODO: make error message more informative
                if (
                    features[rank] is not None
                    and features[rank].shape[0] != incidence_.shape[-1]
                ):
                    raise ValueError("Features have wrong shape.")

        super().__init__(incidence, features)

    def keys(self):
        return list(range(len(self.incidence)))


class HypergraphData(Data):
    def __init__(
        self,
        incidence_hyperedges,
        num_hyperedges,
        incidence_0=None,
        x_0=None,
        x_hyperedges=None,
    ):
        self._hyperedges_key = 1
        incidence = {
            0: incidence_0,
            self._hyperedges_key: incidence_hyperedges,
        }
        features = {
            0: x_0,
            self._hyperedges_key: x_hyperedges,
        }
        super().__init__(incidence, features)
        self.num_hyperedges = num_hyperedges

    def keys(self):
        return [0, self._hyperedges_key]
