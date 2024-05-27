"""Test the message passing module."""
import torch

from topobenchmarkx.data.load.loaders import manual_simple_graph
from topobenchmarkx.transforms.feature_liftings.feature_liftings import (
    ConcatentionLifting,
)
from topobenchmarkx.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)


class TestConcatentionLifting:
    """Test the ConcatentionLifting class."""

    def setup_method(self):
        # Load the graph
        self.data = manual_simple_graph()

        # Initialize a lifting class
        self.lifting = SimplicialCliqueLifting(feature_lifting="concatenation", complex_dim=3)

    def test_lift_features(self):
        # Test the lift_features method
        lifted_data = self.lifting.forward(self.data.clone())

        expected_x1 = torch.tensor(
            [
                [1.0, 5.0],
                [1.0, 10.0],
                [1.0, 100.0],
                [1.0, 5000.0],
                [5.0, 10.0],
                [5.0, 100.0],
                [10.0, 50.0],
                [10.0, 100.0],
                [10.0, 500.0],
                [10.0, 5000.0],
                [50.0, 1000.0],
                [500.0, 1000.0],
                [500.0, 5000.0],
            ]
        )

        expected_x2 = torch.tensor(
            [
                [1.0, 5.0, 1.0, 10.0, 5.0, 10.0],
                [1.0, 5.0, 1.0, 100.0, 5.0, 100.0],
                [1.0, 10.0, 1.0, 100.0, 10.0, 100.0],
                [1.0, 10.0, 1.0, 5000.0, 10.0, 5000.0],
                [5.0, 10.0, 5.0, 100.0, 10.0, 100.0],
                [10.0, 500.0, 10.0, 5000.0, 500.0, 5000.0],
            ]
        )

        expected_x3 = torch.tensor(
            [
                [
                    1.0,
                    5.0,
                    1.0,
                    10.0,
                    5.0,
                    10.0,
                    1.0,
                    5.0,
                    1.0,
                    100.0,
                    5.0,
                    100.0,
                    1.0,
                    10.0,
                    1.0,
                    100.0,
                    10.0,
                    100.0,
                    5.0,
                    10.0,
                    5.0,
                    100.0,
                    10.0,
                    100.0,
                ]
            ]
        )

        assert (
            expected_x1 == lifted_data.x_1
        ).all(), "Something is wrong with the lifted features x_1."
        assert (
            expected_x2 == lifted_data.x_2
        ).all(), "Something is wrong with the lifted features x_2."
        assert (
            expected_x3 == lifted_data.x_3
        ).all(), "Something is wrong with the lifted features x_3."

