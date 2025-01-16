"""Identity transform that does nothing to the input data."""

from topobenchmark.transforms.feature_liftings.base import FeatureLiftingMap


class Identity(FeatureLiftingMap):
    """Identity feature lifting map."""

    # TODO: rename to IdentityFeatureLifting

    def lift_features(self, domain):
        """Lift features of a domain using identity map."""
        return domain
