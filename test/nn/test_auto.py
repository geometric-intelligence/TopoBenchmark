import torch
from .._utils.nn_module_auto_test import NNModuleAutoTest
from topobenchmarkx.nn.backbones.cell.cccn import CCCN
from topobenchmarkx.nn.backbones.cell.cin import CWN
from topobenchmarkx.nn.backbones.hypergraph.edgnn import (
    EDGNN,
    MLP as edgnn_MLP,
    PlainMLP,
    EquivSetConv
)


def test_auto():
    num_nodes = 8
    d_feat = 12
    x = torch.randn(num_nodes, 12)
    edges_1 = torch.randint(0, num_nodes, (2, num_nodes*2))
    edges_2 = torch.randint(0, num_nodes, (2, num_nodes*2))
    
    d_feat_1, d_feat_2 = 5, 17
    hid_channels = 4
    out_channels = 10
    n_layers = 2
    x_1 = torch.randn(num_nodes*2, d_feat_1)
    x_2 = torch.randn(num_nodes*2, d_feat_2)

    auto_test = NNModuleAutoTest([
        {
            "module" : CCCN, 
            "init": (d_feat, ),
            "forward": (x, edges_1, edges_2),
            "assert_shape": (num_nodes, d_feat)
        },
        #{
        #    "module" : CWN, 
        #    "init": (d_feat, d_feat_1, d_feat_2, hid_channels, n_layers),
        #    "forward": (x, x_1, x_2, edges_1, edges_1, edges_1),
        #    #"assert_shape": (num_nodes, d_feat)
        #},
        {
            "module" : EDGNN, 
            "init": (d_feat, ),
            "forward": (x, edges_1),
            "assert_shape": [x.shape]
        },
        {
            "module" : edgnn_MLP, 
            "init": (d_feat, hid_channels, out_channels, 2),
            "forward": (x, ),
            "assert_shape": (num_nodes, out_channels)
        },
        {
            "module" : PlainMLP, 
            "init": (d_feat, hid_channels, out_channels, 2),
            "forward": (x, ),
            "assert_shape": (num_nodes, out_channels)
        },
        #{
        #    "module" : EquivSetConv, 
        #    "init": (d_feat, d_feat),
        #    "forward": (x, torch.arange(num_nodes), edges_1, x),
        #    "assert_shape": (num_nodes, out_channels)
        #},
    ])
    auto_test.run()    
