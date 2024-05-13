"""Test the message passing module."""

import torch

from topobenchmarkx.io.load.loaders import manual_simple_graph
from topobenchmarkx.transforms.feature_liftings.feature_liftings import (
    ProjectionLifting,
)
from topobenchmarkx.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)


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

        expected_x1 = torch.tensor(
            [
                [6.0],
                [11.0],
                [101.0],
                [5001.0],
                [15.0],
                [105.0],
                [60.0],
                [110.0],
                [510.0],
                [5010.0],
                [1050.0],
                [1500.0],
                [5500.0],
            ]
        )

        expected_x2 = torch.tensor(
            [[32.0], [212.0], [222.0], [10022.0], [230.0], [11020.0]]
        )

        expected_x3 = torch.tensor([[696.0]])

        assert (
            expected_x1 == lifted_data.x_1
        ).all(), "Something is wrong with the lifted features x_1."
        assert (
            expected_x2 == lifted_data.x_2
        ).all(), "Something is wrong with the lifted features x_2."
        assert (
            expected_x3 == lifted_data.x_3
        ).all(), "Something is wrong with the lifted features x_3."
