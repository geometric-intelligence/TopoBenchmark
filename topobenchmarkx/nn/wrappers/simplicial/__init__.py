from topobenchmarkx.nn.wrappers.simplicial.san_wrapper import SANWrapper
from topobenchmarkx.nn.wrappers.simplicial.scn_wrapper import SCNWrapper
from topobenchmarkx.nn.wrappers.simplicial.sccnn_wrapper import SCCNNWrapper
from topobenchmarkx.nn.wrappers.simplicial.sccn_wrapper import SCCNWrapper

# ... import other readout classes here
# For example:
# from topobenchmarkx.nn.readouts.other_readout_1 import OtherWrapper1
# from topobenchmarkx.nn.readouts.other_readout_2 import OtherWrapper2

# Export all wrappers and the dictionary
__all__ = [
    "SANWrapper",
    "SCNWrapper",
    "SCCNNWrapper",
    "SCCNWrapper",
    
    # "OtherWrapper1",
    # "OtherWrapper2",
    # ... add other readout classes here
]