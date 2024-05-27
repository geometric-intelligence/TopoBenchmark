"""Test the collate function."""
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

import torch
import torch_geometric

#from topobenchmarkx.data.load.loaders import manual_simple_graph

from topobenchmarkx.transforms.data_manipulations import (
    InfereKNNConnectivity, 
    InfereRadiusConnectivity,
    KeepSelectedDataFields,
    NodeDegrees,
    NodeFeaturesToFloat,
    OneHotDegreeFeatures
)

from topobenchmarkx.utils.config_resolvers import (
    get_default_transform,
    get_monitor_metric,
    get_monitor_mode,
    infer_in_channels,
)

import rootutils

rootutils.setup_root("./", indicator=".project-root", pythonpath=True)

class TestCollateFunction:
    """Test collate_fn."""

    def setup_method(self):
        # Data 1
        x = torch.tensor([
            [2, 2], [2.2, 2], [2.1, 1.5],
            [-3, 2], [-2.7, 2], [-2.5, 1.5],
            [-3, -2], [-2.7, -2], [-2.5, -1.5],
             ])
        self.data = torch_geometric.data.Data(
            x=x,
            num_nodes=len(x),
            field_1 = "some text",
            field_2 = x.clone(),
            preserve_1 = 123,
            preserve_2 = torch.tensor((1, 2, 3))
        )

        # Data 2
        self.data_2 = torch_geometric.data.Data(
            num_nodes = 4,
            x = torch.tensor([[10], [20], [30], [40]]),
            edge_index = torch.tensor([
                [0, 0, 0, 2],
                [1, 2, 3, 3]
            ])
        )

        # Transformations
        self.infere_by_knn = InfereKNNConnectivity(args={"k":3})
        self.infere_by_radius = InfereRadiusConnectivity(args={"r":1.})
        self.keep_selected_fields = KeepSelectedDataFields(base_fields=["x", "num_nodes"], preserved_fields=["preserve_1", "preserve_2"])
        self.node_degress = NodeDegrees(selected_fields=["edge_index"])
        self.node_feature_float = NodeFeaturesToFloat()
        self.one_hot_degree_features = OneHotDegreeFeatures(max_degree=3, degrees_fields="node_degrees", features_fields="one_hot_degree")
    
    def test_infere_connectivity(self):
        data = self.infere_by_knn(self.data.clone())
        assert "edge_index" in data, "No edges in Data object"

    def test_radius_connectivity(self):
        data = self.infere_by_radius(self.data.clone())
        assert "edge_index" in data, "No edges in Data object"
        
    def test_keep_selected_data_fields(self):
        data = self.keep_selected_fields(self.data.clone())
        assert set() == set(data.keys()) - set(self.keep_selected_fields.parameters['base_fields'] + self.keep_selected_fields.parameters['preserved_fields']), \
            "Some fields are not deleted"

    def test_node_degress(self):
        data = self.node_degress(self.data_2.clone())
        expected_degrees = torch.tensor(
            [[3], [0], [1], [0]]
        ).float()
        assert (data.node_degrees == expected_degrees).all(), "Node degrees do not match"

    def test_node_feature_float(self):
        data = self.node_feature_float(self.data_2.clone())
        assert data.x.is_floating_point(), "Node features are not float"

    def test_one_hot_degree_features(self):
        data = self.node_degress(self.data_2.clone())
        data = self.one_hot_degree_features(data)
        expected_vals = torch.tensor([
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ])
        assert (data.one_hot_degree == expected_vals).all()