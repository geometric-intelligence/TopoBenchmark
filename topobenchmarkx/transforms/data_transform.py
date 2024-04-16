# from abc import ABC, abstractmethod

import torch_geometric

from topobenchmarkx.transforms.data_manipulations.manipulations import (
    CalculateSimplicialCurvature,
    DataFieldsToDense,
    EqualGausFeatures,
    IdentityTransform,
    KeepOnlyConnectedComponent,
    NodeDegrees,
    NodeFeaturesToFloat,
    OneHotDegreeFeatures,
)
from topobenchmarkx.transforms.feature_liftings.feature_liftings import (
    ConcatentionLifting,
    ProjectionLifting,
    SetLifting,
)
from topobenchmarkx.transforms.liftings.graph2cell import CellCyclesLifting
from topobenchmarkx.transforms.liftings.graph2hypergraph import (
    HypergraphKHopLifting,
    HypergraphKNearestNeighborsLifting,
)
from topobenchmarkx.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
    SimplicialNeighborhoodLifting,
)

TRANSFORMS = {
    ###
    # Graph -> Hypergraph
    "HypergraphKHopLifting": HypergraphKHopLifting,
    "HypergraphKNearestNeighborsLifting": HypergraphKNearestNeighborsLifting,
    # Graph -> Simplicial Complex
    "SimplicialNeighborhoodLifting": SimplicialNeighborhoodLifting,
    "SimplicialCliqueLifting": SimplicialCliqueLifting,
    # Graph -> Cell Complex
    "CellCyclesLifting": CellCyclesLifting,
    # Feature Liftings
    "ProjectionLifting": ProjectionLifting,
    "ConcatentionLifting": ConcatentionLifting,
    "SetLifting": SetLifting,
    # Data Manipulations
    "Identity": IdentityTransform,
    "DataFieldsToDense": DataFieldsToDense,
    "NodeDegrees": NodeDegrees,
    "OneHotDegreeFeatures": OneHotDegreeFeatures,
    "EqualGausFeatures": EqualGausFeatures,
    "NodeFeaturesToFloat": NodeFeaturesToFloat,
    "CalculateSimplicialCurvature": CalculateSimplicialCurvature,
    "KeepOnlyConnectedComponent": KeepOnlyConnectedComponent,
}


class DataTransform(torch_geometric.transforms.BaseTransform):
    """Abstract class that provides an interface to define a custom data lifting.

    Parameters
    ----------
    transform_name : str
        The name of the transform to be used.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, transform_name, **kwargs):
        super().__init__()

        kwargs["transform_name"] = transform_name
        self.parameters = kwargs

        self.transform = (
            TRANSFORMS[transform_name](**kwargs) if transform_name is not None else None
        )

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Forward pass of the lifting.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data
            The lifted data.
        """
        transformed_data = self.transform(data)
        return transformed_data


if __name__ == "__main__":
    _ = DataTransform()
