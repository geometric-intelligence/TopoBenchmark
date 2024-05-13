from topobenchmarkx.models.readouts.propagate_signal_down import (
    PropagateSignalDown,
)

# ... import other readout classes here
# For example:
# from topobenchmarkx.models.readouts.other_readout_1 import OtherReadout1
# from topobenchmarkx.models.readouts.other_readout_2 import OtherReadout2


# Dictionary of all readouts
READOUTS = {
    "PropagateSignalDown": PropagateSignalDown,
    # "OtherReadout1": OtherReadout1,
    # "OtherReadout2": OtherReadout2,
    # ... add other readout mappings here
}

# Export all readouts and the dictionary
__all__ = [
    "PropagateSignalDown",
    # "OtherReadout1",
    # "OtherReadout2",
    # ... add other readout classes here
    "READOUTS",
]
