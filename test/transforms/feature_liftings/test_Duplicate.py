"""Test the message passing module."""

import torch

from topobenchmarkx.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)


class TestConcatention:
    """Test the Concatention feature lifting class."""

    def setup_method(self):
        """Set up the test."""
        # Initialize a lifting class
        self.lifting = SimplicialCliqueLifting(
            feature_lifting="Duplicate", complex_dim=3
        )

    def test_lift_features(self, simple_graph_0, simple_graph_1):
        """Test the lift_features method.
        
        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            A simple graph data object.
        simple_graph_1 : torch_geometric.data.Data
            A simple graph data object.
        """
        
        data = simple_graph_0
        # Test the lift_features method
        lifted_data = self.lifting.forward(data.clone())
        assert lifted_data.x_2.shape == torch.Size((lifted_data.incidence_2.shape[1], lifted_data.x_0.shape[1]))
        
        data = simple_graph_1
        # Test the lift_features method
        lifted_data = self.lifting.forward(data.clone())

        expected_x1 = torch.ones((lifted_data.incidence_1.shape[1], lifted_data.x_0.shape[1]))
        expected_x2 = torch.ones((lifted_data.incidence_2.shape[1], lifted_data.x_0.shape[1]))
        expected_x3 = torch.ones((lifted_data.incidence_3.shape[1], lifted_data.x_0.shape[1]))

        assert (
            expected_x1 == lifted_data.x_1
        ).all(), "Something is wrong with the lifted features x_1."
        assert (
            expected_x2 == lifted_data.x_2
        ).all(), "Something is wrong with the lifted features x_2."
        assert (
            expected_x3 == lifted_data.x_3
        ).all(), "Something is wrong with the lifted features x_3."

    def test_lift_features(self, simple_graph_0, simple_graph_1):
        """Test the lift_features method.
        
        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            A simple graph data object.
        simple_graph_1 : torch_geometric.data.Data
            A simple graph data object.
        """
        
        data = simple_graph_0
        # Test the lift_features method
        lifted_data = self.lifting.forward(data.clone())
        assert lifted_data.x_2.shape == torch.Size((lifted_data.incidence_2.shape[1], lifted_data.x_0.shape[1]))
        
        data = simple_graph_1
        # Test the lift_features method
        lifted_data = self.lifting.forward(data.clone())

        expected_x1 = torch.ones((lifted_data.incidence_1.shape[1], lifted_data.x_0.shape[1]))
        expected_x2 = torch.ones((lifted_data.incidence_2.shape[1], lifted_data.x_0.shape[1]))
        expected_x3 = torch.ones((lifted_data.incidence_3.shape[1], lifted_data.x_0.shape[1]))

        assert (
            expected_x1 == lifted_data.x_1
        ).all(), "Something is wrong with the lifted features x_1."
        assert (
            expected_x2 == lifted_data.x_2
        ).all(), "Something is wrong with the lifted features x_2."
        assert (
            expected_x3 == lifted_data.x_3
        ).all(), "Something is wrong with the lifted features x_3."
