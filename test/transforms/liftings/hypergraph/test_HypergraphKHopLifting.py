"""Test the message passing module."""

import torch

from topobenchmarkx.dataset.load.loaders import manual_graph
from topobenchmarkx.transforms.liftings.graph2hypergraph import (
    HypergraphKHopLifting,
)


class TestHypergraphKHopLifting:
    """Test the HypergraphKHopLifting class."""

    def setup_method(self):
        # Load the graph
        self.data = manual_graph()

        # Initialise the HypergraphKHopLifting class
        self.lifting_k1 = HypergraphKHopLifting(k_value=1)
        self.lifting_k2 = HypergraphKHopLifting(k_value=2)

    def test_lift_topology(self):
        # Test the lift_topology method
        lifted_data_k1 = self.lifting_k1.forward(self.data.clone())

        expected_n_hyperedges = 9

        expected_incidence_1 = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        assert (
            expected_incidence_1
            == lifted_data_k1.incidence_hyperedges.to_dense()
        ).all(), "Something is wrong with incidence_hyperedges (k=1)."
        assert (
            expected_n_hyperedges == lifted_data_k1.num_hyperedges
        ), "Something is wrong with the number of hyperedges (k=1)."

        lifted_data_k2 = self.lifting_k2.forward(self.data.clone())

        expected_n_hyperedges = 9

        expected_incidence_1 = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            ]
        )

        assert (
            expected_incidence_1
            == lifted_data_k2.incidence_hyperedges.to_dense()
        ).all(), "Something is wrong with incidence_hyperedges (k=2)."
        assert (
            expected_n_hyperedges == lifted_data_k2.num_hyperedges
        ), "Something is wrong with the number of hyperedges (k=2)."
