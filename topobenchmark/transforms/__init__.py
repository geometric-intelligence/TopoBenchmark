"""This module contains the transforms for the topobenchmark package."""

from .data_manipulations import DATA_MANIPULATIONS
from .feature_liftings import FEATURE_LIFTINGS
from .liftings import LIFTINGS

TRANSFORMS = {
    **LIFTINGS,
    **FEATURE_LIFTINGS,
    **DATA_MANIPULATIONS,
}
