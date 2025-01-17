"""Test the message passing module."""

import torch

from topobenchmark.transforms.feature_liftings.projection_sum import (
    ProjectionSum,
)
from topobenchmark.transforms.liftings import (
    Graph2SimplicialLiftingTransform,
    SimplicialCliqueLifting,
)


class TestSimplicialCliqueLifting:
    """Test the SimplicialCliqueLifting class."""

    def setup_method(self):
        # Initialise the SimplicialCliqueLifting class

        lifting_map = SimplicialCliqueLifting(complex_dim=3)
        feature_lifting = ProjectionSum()

        self.lifting_signed = Graph2SimplicialLiftingTransform(
            lifting=lifting_map,
            feature_lifting=feature_lifting,
            signed=True,
        )
        self.lifting_unsigned = Graph2SimplicialLiftingTransform(
            lifting=lifting_map,
            feature_lifting=feature_lifting,
            signed=False,
        )

    def test_lift_topology(self, simple_graph_1):
        """Test the lift_topology method."""

        # Test the lift_topology method
        self.data = simple_graph_1
        lifted_data_signed = self.lifting_signed.forward(self.data.clone())
        lifted_data_unsigned = self.lifting_unsigned.forward(self.data.clone())

        expected_incidence_1 = torch.tensor(
            [
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    -1.0,
                    -1.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            ]
        )

        assert (
            abs(expected_incidence_1)
            == lifted_data_unsigned.incidence_1.to_dense()
        ).all(), (
            "Something is wrong with unsigned incidence_1 (nodes to edges)."
        )
        assert (
            expected_incidence_1 == lifted_data_signed.incidence_1.to_dense()
        ).all(), "Something is wrong with signed incidence_1 (nodes to edges)."

        expected_incidence_2 = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        assert (
            abs(expected_incidence_2)
            == lifted_data_unsigned.incidence_2.to_dense()
        ).all(), "Something is wrong with unsigned incidence_2 (edges to triangles)."
        assert (
            expected_incidence_2 == lifted_data_signed.incidence_2.to_dense()
        ).all(), (
            "Something is wrong with signed incidence_2 (edges to triangles)."
        )

        expected_incidence_3 = torch.tensor(
            [[-1.0], [1.0], [-1.0], [0.0], [1.0], [0.0]]
        )

        assert (
            abs(expected_incidence_3)
            == lifted_data_unsigned.incidence_3.to_dense()
        ).all(), "Something is wrong with unsigned incidence_3 (triangles to tetrahedrons)."
        assert (
            expected_incidence_3 == lifted_data_signed.incidence_3.to_dense()
        ).all(), "Something is wrong with signed incidence_3 (triangles to tetrahedrons)."

    def test_lifted_features_signed(self, simple_graph_1):
        """Test the lift_features method in signed incidence cases."""
        # TODO: can be removed/moved; part of projection sum

        self.data = simple_graph_1
        # Test the lift_features method for signed case
        lifted_data = self.lifting_signed.forward(self.data)

        expected_features_1 = torch.tensor(
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

        assert (
            expected_features_1 == lifted_data.x_1
        ).all(), "Something is wrong with x_1 features."

        expected_features_2 = torch.tensor(
            [[32.0], [212.0], [222.0], [10022.0], [230.0], [11020.0]]
        )

        assert (
            expected_features_2 == lifted_data.x_2
        ).all(), "Something is wrong with x_2 features."

        excepted_features_3 = torch.tensor([[696.0]])

        assert (
            excepted_features_3 == lifted_data.x_3
        ).all(), "Something is wrong with x_3 features."

    def test_lifted_features_unsigned(self, simple_graph_1):
        """Test the lift_features method in unsigned incidence cases."""
        # TODO: redundant. can be moved/removed

        self.data = simple_graph_1
        # Test the lift_features method for unsigned case
        lifted_data = self.lifting_unsigned.forward(self.data)

        expected_features_1 = torch.tensor(
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

        assert (
            expected_features_1 == lifted_data.x_1
        ).all(), "Something is wrong with x_1 features."

        expected_features_2 = torch.tensor(
            [[32.0], [212.0], [222.0], [10022.0], [230.0], [11020.0]]
        )

        assert (
            expected_features_2 == lifted_data.x_2
        ).all(), "Something is wrong with x_2 features."

        excepted_features_3 = torch.tensor([[696.0]])

        assert (
            excepted_features_3 == lifted_data.x_3
        ).all(), "Something is wrong with x_3 features."
