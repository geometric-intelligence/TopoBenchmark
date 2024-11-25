"""This module implements the wrappers for the neural networks."""

from topobenchmarkx.nn.wrappers.base import AbstractWrapper
from topobenchmarkx.nn.wrappers.cell import (
    CANWrapper,
    CCCNWrapper,
    CCXNWrapper,
    CWNWrapper,
)
from topobenchmarkx.nn.wrappers.combinatorial import TuneWrapper
from topobenchmarkx.nn.wrappers.graph import GNNWrapper, GraphMLPWrapper
from topobenchmarkx.nn.wrappers.hypergraph import HypergraphWrapper
from topobenchmarkx.nn.wrappers.simplicial import (
    SANWrapper,
    SCCNNWrapper,
    SCCNWrapper,
    SCNWrapper,
)

# ... import other readout classes here
# For example:
# from topobenchmarkx.nn.wrappers.other_wrapper_1 import OtherWrapper1
# from topobenchmarkx.nn.wrappers.other_wrapper_2 import OtherWrapper2


# Export all wrappers
__all__ = [
    "AbstractWrapper",
    "CANWrapper",
    "CCCNWrapper",
    "CCXNWrapper",
    "CWNWrapper",
    "GNNWrapper",
    "GraphMLPWrapper",
    "HypergraphWrapper",
    "SANWrapper",
    "SCCNNWrapper",
    "SCCNWrapper",
    "SCNWrapper",
    "TuneWrapper",
    # "OtherWrapper1",
    # "OtherWrapper2",
    # ... add other readout classes here
]
