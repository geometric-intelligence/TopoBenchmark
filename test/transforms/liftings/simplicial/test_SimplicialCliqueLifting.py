"""Test the message passing module."""
import pytest
import torch
import rootutils

from topobenchmarkx.transforms.liftings.graph2simplicial import SimplicialCliqueLifting
from topobenchmarkx.io.load.loaders import manual_simple_graph

class TestSimplicialCliqueLifting:
    """Test the SimplicialCliqueLifting class."""
    def setup_method(self):
        # Load the graph
        self.data = manual_simple_graph()
        
        # Initialise the SimplicialCliqueLifting class
        self.lifting_signed = SimplicialCliqueLifting(complex_dim=3, signed=True)
        self.lifting_unsigned = SimplicialCliqueLifting(complex_dim=3, signed=False)
    
    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data_signed = self.lifting_signed.forward(self.data.clone())
        lifted_data_unsigned = self.lifting_unsigned.forward(self.data.clone())

        expected_incidence_1 = torch.tensor(
            [[-1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 1.,  0.,  0.,  0., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  1.,  0.,  0.,  1.,  0., -1., -1., -1., -1.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0., -1.,  0.,  0.],
            [ 0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0., -1., -1.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.],
            [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.]]
        )
        
        assert (abs(expected_incidence_1) == lifted_data_unsigned.incidence_1.to_dense()).all(), "Something is wrong with unsigned incidence_1 (nodes to edges)."
        assert (expected_incidence_1 == lifted_data_signed.incidence_1.to_dense()).all(), "Something is wrong with signed incidence_1 (nodes to edges)."
        
        expected_incidence_2 = torch.tensor(
            [[ 1.,  1.,  0.,  0.,  0.,  0.],
            [-1.,  0.,  1.,  1.,  0.,  0.],
            [ 0., -1., -1.,  0.,  0.,  0.],
            [ 0.,  0.,  0., -1.,  0.,  0.],
            [ 1.,  0.,  0.,  0.,  1.,  0.],
            [ 0.,  1.,  0.,  0., -1.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  1.,  0.,  1.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  1.],
            [ 0.,  0.,  0.,  1.,  0., -1.],
            [ 0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  1.]]
        )

        assert (abs(expected_incidence_2) == lifted_data_unsigned.incidence_2.to_dense()).all(), "Something is wrong with unsigned incidence_2 (edges to triangles)."
        assert (expected_incidence_2 == lifted_data_signed.incidence_2.to_dense()).all(), "Something is wrong with signed incidence_2 (edges to triangles)."

        expected_incidence_3 = torch.tensor(
            [[-1.],
            [ 1.],
            [-1.],
            [ 0.],
            [ 1.],
            [ 0.]]
        )

        assert (abs(expected_incidence_3) == lifted_data_unsigned.incidence_3.to_dense()).all(), "Something is wrong with unsigned incidence_3 (triangles to tetrahedrons)."
        assert (expected_incidence_3 == lifted_data_signed.incidence_3.to_dense()).all(), "Something is wrong with signed incidence_3 (triangles to tetrahedrons)."
    
    def test_lifted_features_signed(self):
        """Test the lift_features method in signed incidence cases."""

        # Test the lift_features method for signed case
        lifted_data = self.lifting_signed.forward(self.data)

        expected_features_1 = torch.tensor(
            [[   4],
            [   9],
            [  99],
            [4999],
            [   5],
            [  95],
            [  40],
            [  90],
            [ 490],
            [4990],
            [ 950],
            [ 500],
            [4500]]
        )

        assert (expected_features_1 == lifted_data.x_1).all(), "Something is wrong with x_1 features."

        expected_features_2 = torch.tensor(
            [[0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.]]
        )

        assert (expected_features_2 == lifted_data.x_2).all(), "Something is wrong with x_2 features."

        excepted_features_3 = torch.tensor(
            [[0.]]
        )

        assert (excepted_features_3 == lifted_data.x_3).all(), "Something is wrong with x_3 features."
    
    def test_lifted_features_unsigned(self):
        """Test the lift_features method in unsigned incidence cases."""

        # Test the lift_features method for unsigned case
        lifted_data = self.lifting_unsigned.forward(self.data)

        expected_features_1 = torch.tensor(
            [[   6.],
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
            [5500.]]
        )

        assert (expected_features_1 == lifted_data.x_1).all(), "Something is wrong with x_1 features."

        expected_features_2 = torch.tensor(
            [[   32.],
            [  212.],
            [  222.],
            [10022.],
            [  230.],
            [11020.]]
        )

        assert (expected_features_2 == lifted_data.x_2).all(), "Something is wrong with x_2 features."

        excepted_features_3 = torch.tensor(
            [[696.]]
        )

        assert (excepted_features_3 == lifted_data.x_3).all(), "Something is wrong with x_3 features."
