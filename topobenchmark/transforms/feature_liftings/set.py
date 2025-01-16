"""Set lifting for r-cell features to r+1-cell features."""

import torch

from topobenchmark.transforms.feature_liftings.base import FeatureLiftingMap


class Set(FeatureLiftingMap):
    """Lift r-cell features to r+1-cells by set operations."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def lift_features(self, domain):
        r"""Concatenate r-cell features to r+1-cell structures.

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
                if rank == 0:
                    values = idxs
                else:
                    values = torch.sort(
                        torch.unique(
                            domain.features[rank][idxs].view(
                                idxs.shape[0], -1
                            ),
                            dim=1,
                        ),
                        dim=1,
                    )[0]
            else:
                values = torch.tensor([])

            domain.update_features(rank + 1, values)

        return domain
