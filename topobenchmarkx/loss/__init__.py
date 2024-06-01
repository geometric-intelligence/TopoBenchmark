from .base import AbstractLoss
from .loss import TBXLoss

# ... import other readout classes here
# For example:
# from topobenchmarkx.loss.other_loss_1 import OtherLoss1
# from topobenchmarkx.loss.other_loss_2 import OtherLoss2

__all__ = [
    "AbstractLoss",
    "TBXLoss"
    # "OtherLoss1",
    # "OtherLoss2",
    # ... add other loss classes here
]