from topobenchmarkx.transforms.data_manipulations import IdentityTransform  # noqa: I001
from topobenchmarkx.transforms.feature_liftings.concatenation import Concatenation
from topobenchmarkx.transforms.feature_liftings.projection_sum import ProjectionSum
from topobenchmarkx.transforms.feature_liftings.set import Set


FEATURE_LIFTINGS = {
    "Concatenation": Concatenation,
    "ProjectionSum": ProjectionSum,
    "Set": Set,
    None: IdentityTransform,
}

__all__ = [
    "Concatenation",
    "ProjectionSum",
    "Set",
    "FEATURE_LIFTINGS",
]
