"""This module implements the encoders for the neural networks."""

from .all_cell_encoder import AllCellFeatureEncoder  # noqa: F401
from .base import AbstractFeatureEncoder  # noqa: F401

# ... import other encoders classes here
# For example:
# from topobenchmarkx.nn.encoders.other_encoder_1 import OtherEncoder1
# from topobenchmarkx.nn.encoders.other_encoder_2 import OtherEncoder2

__all__ = [
    "AbstractFeatureEncoder" "AllCellFeatureEncoder"
    # "OtherEncoder1",
    # "OtherEncoder2",
    # ... add other readout classes here
]
