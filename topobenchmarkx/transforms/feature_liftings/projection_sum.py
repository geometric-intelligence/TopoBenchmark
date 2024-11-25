"""ProjectionSum class."""

from .base import FeatureLiftingMap


class ProjectionSum(FeatureLiftingMap):
    r"""Lift r-cell features to r+1-cells by projection."""

    def lift_features(self, domain):
        r"""Project r-cell features of a graph to r+1-cell structures.

        Parameters
        ----------
        data : PlainComplex
            The input data to be lifted.

        Returns
        -------
        PlainComplex
            Domain with the lifted features.
        """
        for rank in range(domain.max_rank - 1):
            if domain.features[rank + 1] is not None:
                continue

            domain.features[rank + 1] = domain.propagate_values(
                rank, domain.features[rank]
            )

        return domain
