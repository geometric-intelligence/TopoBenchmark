from topobenchmarkx.transforms.data_manipulations.calculate_simplicial_curvature import CalculateSimplicialCurvature  # noqa: I001
from topobenchmarkx.transforms.data_manipulations.equal_gaus_features import EqualGausFeatures
from topobenchmarkx.transforms.data_manipulations.identity_transform import IdentityTransform
from topobenchmarkx.transforms.data_manipulations.infere_knn_connectivity import InfereKNNConnectivity
from topobenchmarkx.transforms.data_manipulations.infere_radius_connectivity import InfereRadiusConnectivity
from topobenchmarkx.transforms.data_manipulations.keep_only_connected_component import KeepOnlyConnectedComponent
from topobenchmarkx.transforms.data_manipulations.keep_selected_data_fields import KeepSelectedDataFields
from topobenchmarkx.transforms.data_manipulations.node_degrees import NodeDegrees
from topobenchmarkx.transforms.data_manipulations.node_features_to_float import NodeFeaturesToFloat
from topobenchmarkx.transforms.data_manipulations.one_hot_degree_features import OneHotDegreeFeatures

DATA_MANIPULATIONS = {
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
    "IdentityTransform",
    "InfereKNNConnectivity",
    "InfereRadiusConnectivity",
    "EqualGausFeatures",
    "NodeFeaturesToFloat",
    "NodeDegrees",
    "KeepOnlyConnectedComponent",
    "CalculateSimplicialCurvature",
    "OneHotDegreeFeatures",
    "KeepSelectedDataFields",
    "DATA_MANIPULATIONS",
]
