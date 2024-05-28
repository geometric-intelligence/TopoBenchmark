"""Test the collate function."""
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

import torch
import torch_geometric


from topobenchmarkx.transforms.data_manipulations import (
    InfereKNNConnectivity, 
    InfereRadiusConnectivity,
    KeepSelectedDataFields
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
        """Setup the test.
        
        For this test we load the MUTAG dataset.
        
        Parameters
        ----------
        None
        """
        """
        OmegaConf.register_new_resolver("get_default_transform", get_default_transform)
        OmegaConf.register_new_resolver("get_monitor_metric", get_monitor_metric)
        OmegaConf.register_new_resolver("get_monitor_mode", get_monitor_mode)
        OmegaConf.register_new_resolver("infer_in_channels", infer_in_channels)
        OmegaConf.register_new_resolver(
            "parameter_multiplication", lambda x, y: int(int(x) * int(y))
        )
        
        initialize(version_base="1.3", config_path="../../configs", job_name="job")
        cfg = compose(config_name="train.yaml", overrides=["dataset=PROTEINS_TU"])
        
        graph_loader = hydra.utils.instantiate(cfg.dataset, _recursive_=False)

        datasets = graph_loader.load()
        self.batch_size = 2
        datamodule = DefaultDataModule(
            dataset_train=datasets[0],
            dataset_val=datasets[1],
            dataset_test=datasets[2],
            batch_size=self.batch_size
        )
        self.val_dataloader = datamodule.val_dataloader()
        self.val_dataset = datasets[1]
        """
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
        # Data Manipulations
        self.infere_by_knn = InfereKNNConnectivity(args={"k":3})
        self.infere_by_radius = InfereRadiusConnectivity(args={"r":1.})
        self.keep_selected_fields = KeepSelectedDataFields(base_fields=["x", "num_nodes"], preserved_fields=["preserve_1", "preserve_2"])
    
    
    def test_infere_connectivity(self):
        data = self.infere_by_knn(self.data.clone())
        assert "edge_index" in data, "No edges in Data object"

    def test_radius_connectivity(self):
        data = self.infere_by_radius(self.data.clone())
        assert "edge_index" in data, "No edges in Data object"
        
    #def test_keep_selected_data_fields(self):
    #    orig_data = self.data.clone()
    #    data = self.keep_selected_fields(orig_data)
    #    assert 0
