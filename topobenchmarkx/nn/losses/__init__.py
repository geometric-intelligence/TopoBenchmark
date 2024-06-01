from topobenchmarkx.nn.losses.base import AbstractLoss
from topobenchmarkx.nn.losses.default_loss import DefaultLoss

# ... import other readout classes here
# For example:
# from topobenchmarkx.nn.losses.other_loss_1 import OtherLoss1
# from topobenchmarkx.nn.losses.other_loss_2 import OtherLoss2

__all__ = [
    "AbstractLoss",
    "DefaultLoss"
    # "OtherLoss1",
    # "OtherLoss2",
    # ... add other loss classes here
]