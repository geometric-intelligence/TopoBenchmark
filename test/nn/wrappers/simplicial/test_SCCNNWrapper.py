"""Unit tests for SCCNNWrapper"""

import torch
from torch_geometric.utils import get_laplacian
from ...._utils.nn_module_auto_test import NNModuleAutoTest
from ...._utils.flow_mocker import FlowMocker
from topobenchmarkx.nn.backbones.simplicial import SCCNNCustom
from topobenchmarkx.nn.wrappers import SCCNNWrapper
from topobenchmarkx.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)

class TestSCCNNWrapper:
    def test_call(self, sg1_clique_lifted):
        data = sg1_clique_lifted

        out_dim = 4
        conv_order = 1
        sc_order = 3
        init_args = (data.x_0.shape[1], data.x_1.shape[1], data.x_2.shape[1]), (out_dim, out_dim, out_dim), conv_order, sc_order

        wrapper = SCCNNWrapper(
            SCCNNCustom(*init_args), 
            out_channels=out_dim, 
            num_cell_dimensions=3
        )
        out = wrapper(data)
        # Assert keys in output
        for key in ["labels", "batch_0", "x_0", "x_1", "x_2"]:
            assert key in out
