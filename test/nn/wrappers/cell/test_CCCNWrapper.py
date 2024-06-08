"""Unit tests for SCCNNWrapper"""

import torch
from torch_geometric.utils import get_laplacian
from ...._utils.nn_module_auto_test import NNModuleAutoTest
from ...._utils.flow_mocker import FlowMocker
from topobenchmarkx.nn.wrappers import (
    AbstractWrapper,
    CCCNWrapper
)
from topobenchmarkx.nn.backbones.cell.cccn import CCCN
from unittest.mock import MagicMock


class TestCCCNWrapper:
    def test_call(self, mocker, sg1_clique_lifted):
        data = sg1_clique_lifted

        out_channels = data.x_1.shape[1]
        num_cell_dimensions = 2

        wrapper = CCCNWrapper(
            CCCN(
                data.x_1.shape[1]
            ), 
            out_channels=out_channels, 
            num_cell_dimensions=num_cell_dimensions
        )
        out = wrapper(data)

        for key in ["labels", "batch_0", "x_0", "x_1"]:
            assert key in out
        
