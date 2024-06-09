"""Unit tests for config resolvers"""

import pytest
from omegaconf import OmegaConf
from topobenchmarkx.utils.config_resolvers import (
    infer_in_channels,
    get_default_transform,
    get_monitor_metric,
    get_monitor_mode
)


class TestConfigResolvers:

    def setup_method(self):
        self.dataset_config = OmegaConf.load("configs/dataset/graph/MUTAG.yaml")
        self.cliq_lift_transform = OmegaConf.load("configs/transforms/liftings/graph2simplicial/clique.yaml")
        self.feature_lift_transform = OmegaConf.load("configs/transforms/feature_liftings/concatenate.yaml")

    def test_get_default_transform(self):
        out = get_default_transform("graph/MUTAG", "graph/gat")
        assert out == "no_transform"

        out = get_default_transform("graph/MUTAG", "cell/can")
        assert out == "liftings/graph2cell_default"

        with pytest.raises(ValueError, match="Invalid combination") as e:
            get_default_transform("graph/MUTAG", "combinatorial/some")
    
    def test_get_monitor_metric(self):
        out = get_monitor_metric("classification", "F1")
        assert out == "val/F1" 

    def test_get_monitor_mode(self):
        out = get_monitor_mode("regression")
        assert out == "min"

    def test_infer_in_channels(self):
        in_channels = infer_in_channels(self.dataset_config, self.cliq_lift_transform)
        assert in_channels == [7]
