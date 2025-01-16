"""Concatenation feature lifting."""

import torch

from topobenchmark.transforms.feature_liftings.base import FeatureLiftingMap


class Concatenation(FeatureLiftingMap):
    """Lift r-cell features to r+1-cells by concatenation."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def lift_features(self, domain):
        r"""Concatenate r-cell features to obtain r+1-cell features.

        Parameters
        ----------
        data : Complex
            The input data to be lifted.

        Returns
        -------
        Complex
            Domain with the lifted features.
        """
        for rank in range(domain.max_rank - 1):
            if domain.features[rank + 1] is not None:
                continue

            # TODO: different if hyperedges?
            idx_to_project = rank

            incidence = domain.incidence[rank + 1]
            _, n = incidence.shape

            if n != 0:
                idxs_list = []
                for n_feature in range(n):
                    idxs_for_feature = incidence.indices()[
                        0, incidence.indices()[1, :] == n_feature
                    ]
                    idxs_list.append(torch.sort(idxs_for_feature)[0])

                idxs = torch.stack(idxs_list, dim=0)
                values = domain.features[idx_to_project][idxs].view(n, -1)
            else:
                m = domain.features[rank].shape[1] * (rank + 2)
                values = torch.zeros([0, m])

            domain.update_features(rank + 1, values)

        return domain
