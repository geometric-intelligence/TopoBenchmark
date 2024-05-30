from topobenchmarkx.nn.readouts.readout import AbstractZeroCellReadOut
from topobenchmarkx.nn.readouts.propagate_signal_down import PropagateSignalDown
from topobenchmarkx.nn.readouts.identical import NoReadOut

# ... import other readout classes here
# For example:
# from topobenchmarkx.nn.readouts.other_readout_1 import OtherReadout1
# from topobenchmarkx.nn.readouts.other_readout_2 import OtherReadout2

# Export all readouts and the dictionary
__all__ = [
    "AbstractZeroCellReadOut",
    "PropagateSignalDown",
    "NoReadOut"
    # "OtherReadout1",
    # "OtherReadout2",
    # ... add other readout classes here
]
