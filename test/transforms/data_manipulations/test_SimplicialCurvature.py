"""Test simplicial curvature calculation."""

import torch
from torch_geometric.data import Data

from topobenchmark.data.utils import (
    Complex2Dict,
    Data2NxGraph,
    TnxComplex2Complex,
)
from topobenchmark.transforms.data_manipulations import (
    CalculateSimplicialCurvature,
)
from topobenchmark.transforms.liftings import (
    LiftingTransform,
    SimplicialCliqueLifting,
)


class TestSimplicialCurvature:
    """Test simplicial curvature calculation."""

    def test_simplicial_curvature(self, simple_graph_1):
        """Test simplicial curvature calculation.

        Parameters
        ----------
        simple_graph_1 : torch_geometric.data.Data
            A simple graph fixture.
        """
        simplicial_curvature = CalculateSimplicialCurvature()

        lifting_unsigned = LiftingTransform(
            lifting=SimplicialCliqueLifting(complex_dim=3),
            data2domain=Data2NxGraph(),
            domain2domain=TnxComplex2Complex(signed=False),
            domain2dict=Complex2Dict(),
        )

        data = lifting_unsigned(simple_graph_1)
        data["0_cell_degrees"] = torch.unsqueeze(
            torch.sum(data["incidence_1"], dim=1).to_dense(), dim=1
        )
        data["1_cell_degrees"] = torch.unsqueeze(
            torch.sum(data["incidence_2"], dim=1).to_dense(), dim=1
        )
        data["2_cell_degrees"] = torch.unsqueeze(
            torch.sum(data["incidence_3"], dim=1).to_dense(), dim=1
        )

        res = simplicial_curvature(data)
        assert isinstance(res, Data)
