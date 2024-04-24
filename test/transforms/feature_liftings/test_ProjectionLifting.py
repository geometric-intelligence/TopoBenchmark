"""Test the message passing module."""
import pytest
import torch
import rootutils


from topobenchmarkx.transforms.feature_liftings.feature_liftings import ProjectionLifting
from topobenchmarkx.transforms.liftings.graph2simplicial import SimplicialCliqueLifting
from topobenchmarkx.io.load.loaders import manual_simple_graph

class TestProjectionLifting:
    """Test the ProjectionLifting class."""
    def setup_method(self):
        # Load the graph
        self.data = manual_simple_graph()
        
        # Initialize a lifting class 
        self.lifting = SimplicialCliqueLifting(complex_dim=3)
        # Initialize the ProjectionLifting class
        self.feature_lifting = ProjectionLifting()
    
    def test_lift_features(self):
        # Test the lift_features method
        lifted_data = self.lifting.forward(self.data.clone())
        del lifted_data.x_1
        del lifted_data.x_2
        del lifted_data.x_3
        lifted_data = self.feature_lifting.forward(lifted_data)
        
        expected_x1 = torch.tensor([[   6.],
                                    [  11.],
                                    [ 101.],
                                    [5001.],
                                    [  15.],
                                    [ 105.],
                                    [  60.],
                                    [ 110.],
                                    [ 510.],
                                    [5010.],
                                    [1050.],
                                    [1500.],
                                    [5500.]])
        
        expected_x2 = torch.tensor([[   32.],
                                    [  212.],
                                    [  222.],
                                    [10022.],
                                    [  230.],
                                    [11020.]])
        
        expected_x3 = torch.tensor([[696.]])
        
        assert (expected_x1==lifted_data.x_1).all(), "Something is wrong with the lifted features x_1."
        assert (expected_x2==lifted_data.x_2).all(), "Something is wrong with the lifted features x_2."
        assert (expected_x3==lifted_data.x_3).all(), "Something is wrong with the lifted features x_3."
        