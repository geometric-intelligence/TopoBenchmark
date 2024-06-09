"""Unit tests for EDGNN"""

import torch
from ...._utils.nn_module_auto_test import NNModuleAutoTest
from topobenchmarkx.nn.backbones.hypergraph.edgnn import (
    EDGNN,
    MLP as edgnn_MLP,
    PlainMLP,
    EquivSetConv,
    JumpLinkConv,
    MeanDegConv
)


def test_EDGNN(random_graph_input):
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    auto_test = NNModuleAutoTest([
        {
            "module" : EDGNN, 
            "init": (x.shape[1], ),
            "forward": (x, edges_1),
            "assert_shape": x.shape
        },
    ])
    auto_test.run()


def test_edgnn_MLP(random_graph_input):
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    hid_channels = 4
    out_channels = 10
    num_layers = 2

    auto_test = NNModuleAutoTest([
        {
            "module" : edgnn_MLP, 
            "init": (x.shape[1], hid_channels, out_channels, num_layers),
            "forward": (x, ),
            "assert_shape": (x.shape[0], out_channels)
        },
    ])
    auto_test.run()


def test_PlainMLP(random_graph_input):
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    num_nodes = x.shape[0]
    hid_channels = 4
    out_channels = 10
    num_layers = 2

    auto_test = NNModuleAutoTest([
        {
            "module" : PlainMLP, 
            "init": (x.shape[1], hid_channels, out_channels, num_layers),
            "forward": (x, ),
            "assert_shape": (num_nodes, out_channels)
        },
    ])
    auto_test.run()


def test_EquivSetConv(random_graph_input):
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    num_nodes = x.shape[0]
    hid_channels = 4
    out_channels = 10
    num_layers = 2

    auto_test = NNModuleAutoTest([
        {
            "module" : EquivSetConv, 
            "init": (x.shape[1], x.shape[1]),
            "forward":  (x, edges_1[0], edges_1[1], x),
            "assert_shape": x.shape
        },
    ])
    auto_test.run()


def test_JumpLinkConv(random_graph_input):
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    num_nodes = x.shape[0]
    hid_channels = 4
    out_channels = 10
    num_layers = 2

    auto_test = NNModuleAutoTest([
        {
            "module" : JumpLinkConv, 
            "init": (x.shape[1], x.shape[1]),
            "forward":  (x, edges_1[0], edges_1[1], x),
            "assert_shape": x.shape
        },
    ])
    auto_test.run()

def test_JumpLinkConv(random_graph_input):
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    num_nodes = x.shape[0]
    hid_channels = 4
    out_channels = 10
    num_layers = 2

    auto_test = NNModuleAutoTest([
        {
            "module" : JumpLinkConv, 
            "init": (x.shape[1], x.shape[1]),
            "forward":  (x, edges_1[0], edges_1[1], x),
            "assert_shape": x.shape
        },
    ])
    auto_test.run()

