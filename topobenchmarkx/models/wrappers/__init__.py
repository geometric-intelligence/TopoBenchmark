from topobenchmarkx.models.wrappers.wrapper import DefaultWrapper
from topobenchmarkx.models.wrappers.graph import GNNWrapper
from topobenchmarkx.models.wrappers.hypergraph import HypergraphWrapper
from topobenchmarkx.models.wrappers.simplicial import SANWrapper, SCNWrapper, SCCNNWrapper, SCCNWrapper
from topobenchmarkx.models.wrappers.cell import CANWrapper, CWNDCMWrapper, CWNWrapper, CCXNWrapper

# ... import other readout classes here
# For example:
# from topobenchmarkx.models.wrappers.other_wrapper_1 import OtherWrapper1
# from topobenchmarkx.models.wrappers.other_wrapper_2 import OtherWrapper2


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
    "CWNDCMWrapper",
    "CWNWrapper",
    "CCXNWrapper",
    # "OtherWrapper1",
    # "OtherWrapper2",
    # ... add other readout classes here
]