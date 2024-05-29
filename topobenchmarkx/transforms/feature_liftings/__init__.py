from .concatenation import Concatenation
from .identity import Identity
from .projection_sum import ProjectionSum
from .set import Set

FEATURE_LIFTINGS = {
    "Concatenation": Concatenation,
    "ProjectionSum": ProjectionSum,
    "Set": Set,
    None: Identity,
}

__all__ = [
    "Concatenation",
    "ProjectionSum",
    "Set",
    "FEATURE_LIFTINGS",
]
