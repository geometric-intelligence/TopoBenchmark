"""ProjectionSum class."""

import torch

from topobenchmark.transforms.feature_liftings.base import FeatureLiftingMap


class ProjectionSum(FeatureLiftingMap):
    r"""Lift r-cell features to r+1-cells by projection."""

    def lift_features(self, domain):
        r"""Project r-cell features of a graph to r+1-cell structures.

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

            domain.update_features(
                rank + 1,
                torch.matmul(
                    torch.abs(domain.incidence[rank + 1].t()),
                    domain.features[rank],
                ),
            )

        return domain
