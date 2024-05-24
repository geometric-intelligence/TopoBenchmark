from topobenchmarkx.nn.wrappers.wrapper import DefaultWrapper
from topobenchmarkx.nn.wrappers.graph import GNNWrapper
from topobenchmarkx.nn.wrappers.hypergraph import HypergraphWrapper
from topobenchmarkx.nn.wrappers.simplicial import SANWrapper, SCNWrapper, SCCNNWrapper, SCCNWrapper
from topobenchmarkx.nn.wrappers.cell import CANWrapper, CCCNWrapper, CWNWrapper, CCXNWrapper

# ... import other readout classes here
# For example:
# from topobenchmarkx.nn.wrappers.other_wrapper_1 import OtherWrapper1
# from topobenchmarkx.nn.wrappers.other_wrapper_2 import OtherWrapper2


# Export all wrappers
__all__ = [
    "DefaultWrapper",
    "GNNWrapper",
    "HypergraphWrapper",
    "SANWrapper",
    "SCNWrapper",
    "SCCNNWrapper",
    "SCCNWrapper",
    "CANWrapper",
    "CCCNWrapper",
    "CWNWrapper",
    "CCXNWrapper",
    # "OtherWrapper1",
    # "OtherWrapper2",
    # ... add other readout classes here
]