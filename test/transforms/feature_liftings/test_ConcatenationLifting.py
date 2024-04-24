"""Test the message passing module."""
import pytest
import torch
import rootutils


from topobenchmarkx.transforms.feature_liftings.feature_liftings import ConcatentionLifting
from topobenchmarkx.transforms.liftings.graph2simplicial import SimplicialCliqueLifting
from topobenchmarkx.io.load.loaders import manual_simple_graph

class TestConcatentionLifting:
    """Test the ConcatentionLifting class."""
    def setup_method(self):
        # Load the graph
        self.data = manual_simple_graph()
        
        # Initialize a lifting class 
        self.lifting = SimplicialCliqueLifting(complex_dim=3)
        # Initialize the ConcatentionLifting class
        self.feature_lifting = ConcatentionLifting()
    
    def test_lift_features(self):
        # Test the lift_features method
        lifted_data = self.lifting.forward(self.data.clone())
        del lifted_data.x_1
        del lifted_data.x_2
        del lifted_data.x_3
        lifted_data = self.feature_lifting.forward(lifted_data)
        
        expected_x1 = torch.tensor([[   1.,    5.],
                                    [   1.,   10.],
                                    [   1.,  100.],
                                    [   1., 5000.],
                                    [   5.,   10.],
                                    [   5.,  100.],
                                    [  10.,   50.],
                                    [  10.,  100.],
                                    [  10.,  500.],
                                    [  10., 5000.],
                                    [  50., 1000.],
                                    [ 500., 1000.],
                                    [ 500., 5000.]])
        
        
        expected_x2 = torch.tensor([[   1.,    5.,     1.,   10.,    5.,   10.],
                                    [   1.,    5.,     1.,  100.,    5.,  100.],
                                    [   1.,   10.,     1.,  100.,   10.,  100.],
                                    [   1.,   10.,     1., 5000.,   10., 5000.],
                                    [   5.,   10.,     5.,  100.,   10.,  100.],
                                    [  10.,  500.,    10., 5000.,  500., 5000.]])

        expected_x3 = torch.tensor([[  1.,   5.,   1.,  10.,   5.,  10.,   
                                       1.,   5.,   1., 100.,   5., 100.,
                                       1.,  10.,   1., 100.,  10., 100.,
                                       5.,  10.,   5., 100.,  10., 100.]])
        
        assert (expected_x1==lifted_data.x_1).all(), "Something is wrong with the lifted features x_1."
        assert (expected_x2==lifted_data.x_2).all(), "Something is wrong with the lifted features x_2."
        assert (expected_x3==lifted_data.x_3).all(), "Something is wrong with the lifted features x_3."
        