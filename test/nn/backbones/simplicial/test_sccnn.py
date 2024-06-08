"""Unit tests for CCCN"""

import torch
from torch_geometric.utils import get_laplacian
from ...._utils.nn_module_auto_test import NNModuleAutoTest
from topobenchmarkx.nn.backbones.simplicial import SCCNNCustom


"""
def test_SCCNNCustom(random_graph_input):
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    L = get_laplacian(edges_1)[1]
    b1 = torch.randint(0, 2, (x.shape[0], x_1.shape[0]))
    b2 = torch.randint(0, 2, (x_1.shape[0], x_2.shape[0]))
    #assert 0
    #x_1_to_0_upper = torch.mm(b1, x_1)
    #x_0_1_lower = torch.mm(b1.T, x_0)

    #x_2_1_upper = torch.mm(b2, x_2)
    #x_1_2_lower = torch.mm(b2.T, x_1)

    conv_order = 1
    sc_order = 2

    auto_test = NNModuleAutoTest([
        {
            "module" : SCCNNCustom, 
            "init": ((x.shape[1], x_1.shape[1], x_2.shape[1]), (x.shape[1], x_1.shape[1], x_2.shape[1]), conv_order, sc_order),
            "forward":  ((x, x_1, x_2), (L, L, L, L), (b1, b2)),
            "assert_shape": x.shape
        },
    ])
    auto_test.run()
"""