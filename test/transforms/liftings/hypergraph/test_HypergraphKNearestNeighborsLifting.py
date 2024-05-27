"""Test the message passing module."""

import torch

from topobenchmarkx.data.load.loaders import manual_graph
from topobenchmarkx.transforms.liftings.graph2hypergraph import (
    HypergraphKNNLifting,
)


class TestHypergraphKNNLifting:
    """Test the HypergraphKNNLifting class."""

    def setup_method(self):
        # Load the graph
        self.data = manual_graph()

        # Initialise the HypergraphKNNLifting class
        self.lifting_k2 = HypergraphKNNLifting(
            k_value=2, loop=True
        )
        self.lifting_k3 = HypergraphKNNLifting(
            k_value=3, loop=True
        )

    def test_lift_topology(self):
        # Test the lift_topology method
        lifted_data_k2 = self.lifting_k2.forward(self.data.clone())

        expected_n_hyperedges = 9

        expected_incidence_1 = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            ]
        )

        assert (
            expected_incidence_1
            == lifted_data_k2.incidence_hyperedges.to_dense()
        ).all(), "Something is wrong with incidence_hyperedges (k=2)."
        assert (
            expected_n_hyperedges == lifted_data_k2.num_hyperedges
        ), "Something is wrong with the number of hyperedges (k=2)."

        lifted_data_k3 = self.lifting_k3.forward(self.data.clone())

        expected_n_hyperedges = 9

        expected_incidence_1 = torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ]
        )

        assert (
            expected_incidence_1
            == lifted_data_k3.incidence_hyperedges.to_dense()
        ).all(), "Something is wrong with incidence_hyperedges (k=3)."
        assert (
            expected_n_hyperedges == lifted_data_k3.num_hyperedges
        ), "Something is wrong with the number of hyperedges (k=3)."
