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
from topobenchmarkx.transforms.liftings.graph2cell import (
    CellCycleLifting,
)
from topobenchmarkx.transforms.liftings.graph2hypergraph import (
    HypergraphKHopLifting,
    HypergraphKNNLifting,
)
from topobenchmarkx.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
    SimplicialKHopLifting,
)

# Dictionary of all available transforms
TRANSFORMS = {
    # Graph -> Hypergraph
    "HypergraphKHopLifting": HypergraphKHopLifting,
    "HypergraphKNNLifting": HypergraphKNNLifting,
    # Graph -> Simplicial Complex
    "SimplicialKHopLifting": SimplicialKHopLifting,
    "SimplicialCliqueLifting": SimplicialCliqueLifting,
    # Graph -> Cell Complex
    "CellCycleLifting": CellCycleLifting,
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


FEATURE_LIFTINGS = {
    "ProjectionSum": ProjectionSum,
    "ConcatentionLifting": ConcatentionLifting,
    "SetLifting": SetLifting,
    None: IdentityTransform,
}


__all__ = [
    "TRANSFORMS",
    "FEATURE_LIFTINGS"
]
