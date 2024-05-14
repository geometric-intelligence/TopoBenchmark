# Data manipulation transforms
from topobenchmarkx.transforms.data_manipulations import (
    CalculateSimplicialCurvature,
    EqualGausFeatures,
    IdentityTransform,
    InfereKNNConnectivity,
    InfereRadiusConnectivity,
    KeepOnlyConnectedComponent,
    KeepSelectedDataFields,
    NodeDegrees,
    NodeFeaturesToFloat,
    OneHotDegreeFeatures,
)

# Feature liftings
from topobenchmarkx.transforms.feature_liftings import (
    ConcatentionLifting,
    ProjectionSum,
    SetLifting,
)

# Topology Liftings
from topobenchmarkx.transforms.liftings import (
    CellCyclesLifting,
    HypergraphKHopLifting,
    HypergraphKNearestNeighborsLifting,
    SimplicialCliqueLifting,
    SimplicialNeighborhoodLifting,
)

# Dictionalry of all available transforms
TRANSFORMS = {
    # Graph -> Hypergraph
    "HypergraphKHopLifting": HypergraphKHopLifting,
    "HypergraphKNearestNeighborsLifting": HypergraphKNearestNeighborsLifting,
    # Graph -> Simplicial Complex
    "SimplicialNeighborhoodLifting": SimplicialNeighborhoodLifting,
    "SimplicialCliqueLifting": SimplicialCliqueLifting,
    # Graph -> Cell Complex
    "CellCyclesLifting": CellCyclesLifting,
    # Feature Liftings
    "ProjectionSum": ProjectionSum,
    "ConcatentionLifting": ConcatentionLifting,
    "SetLifting": SetLifting,
    # Data Manipulations
    "Identity": IdentityTransform,
    "InfereKNNConnectivity": InfereKNNConnectivity,
    "InfereRadiusConnectivity": InfereRadiusConnectivity,
    "NodeDegrees": NodeDegrees,
    "OneHotDegreeFeatures": OneHotDegreeFeatures,
    "EqualGausFeatures": EqualGausFeatures,
    "NodeFeaturesToFloat": NodeFeaturesToFloat,
    "CalculateSimplicialCurvature": CalculateSimplicialCurvature,
    "KeepOnlyConnectedComponent": KeepOnlyConnectedComponent,
    "KeepSelectedDataFields": KeepSelectedDataFields,
}



__all__ = [
    "TRANSFORMS",
]