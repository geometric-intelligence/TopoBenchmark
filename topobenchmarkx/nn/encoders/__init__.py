"""This module implements the encoders for the neural networks."""

from .all_cell_encoder import AllCellFeatureEncoder
from .base import AbstractFeatureEncoder
from .sann_encoder import SANNFeatureEncoder

# ... import other encoders classes here
# For example:
# from topobenchmarkx.nn.encoders.other_encoder_1 import OtherEncoder1
# from topobenchmarkx.nn.encoders.other_encoder_2 import OtherEncoder2

__all__ = [
    "AbstractFeatureEncoder",
    "AllCellFeatureEncoder",
    "SANNFeatureEncoder",
    # "OtherEncoder1",
    # "OtherEncoder2",
    # ... add other readout classes here
]
