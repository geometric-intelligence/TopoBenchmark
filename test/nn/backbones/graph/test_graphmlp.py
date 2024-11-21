"""Unit tests for GraphMLP."""

from topobenchmarkx.nn.backbones.graph.graph_mlp import GraphMLP

def testGraphMLP(random_graph_input):
    """ Unit test for GraphMLP.
    
    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of input tensors for testing EDGNN.
    """
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    model = GraphMLP(x.shape[1], x.shape[1])
    out = model(x)
    assert out[0].shape == x.shape
    assert list(out[1].shape) == [8,8]
