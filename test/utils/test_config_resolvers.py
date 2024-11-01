"""Unit tests for config resolvers."""

import pytest
from omegaconf import OmegaConf
from topobenchmarkx.utils.config_resolvers import (
    infer_in_channels,
    infere_num_cell_dimensions,
    get_default_metrics,
    get_default_transform,
    get_monitor_metric,
    get_monitor_mode,
    get_required_lifting,
)


class TestConfigResolvers:
    """Test config resolvers."""

    def setup_method(self):
        """Setup method."""
        self.dataset_config = OmegaConf.load("configs/dataset/graph/MUTAG.yaml")
        self.cliq_lift_transform = OmegaConf.load("configs/transforms/liftings/graph2simplicial/clique.yaml")
        self.feature_lift_transform = OmegaConf.load("configs/transforms/feature_liftings/concatenate.yaml")
        
    def test_get_default_metrics(self):
        """Test get_default_metrics."""
        out = get_default_metrics("classification")
        assert out == ["accuracy", "precision", "recall", "auroc"]

        out = get_default_metrics("regression")
        assert out == ["mse", "mae"]

        with pytest.raises(ValueError, match="Invalid task") as e:
            get_default_metrics("some_task")

    def test_get_default_transform(self):
        """Test get_default_transform."""
        out = get_default_transform("graph/MUTAG", "graph/gat")
        assert out == "no_transform"

        out = get_default_transform("graph/MUTAG", "cell/can")
        assert out == "liftings/graph2cell_default"

        out = get_default_transform("graph/ZINC", "cell/can")
        assert out == "dataset_defaults/ZINC"
        
    def test_get_required_lifting(self):
        """Test get_required_lifting."""
        out = get_required_lifting("graph", "graph/gat")
        assert out == "no_lifting"

        out = get_required_lifting("graph", "cell/can")
        assert out == "graph2cell_default"
    
    def test_get_monitor_metric(self):
        """Test get_monitor_metric."""
        out = get_monitor_metric("classification", "F1")
        assert out == "val/F1" 
        
        with pytest.raises(ValueError, match="Invalid task") as e:
            get_monitor_metric("mix", "F1")

    def test_get_monitor_mode(self):
        """Test get_monitor_mode."""
        out = get_monitor_mode("regression")
        assert out == "min"
        
        out = get_monitor_mode("classification")
        assert out == "max"
        
        with pytest.raises(ValueError, match="Invalid task") as e:
            get_monitor_mode("mix")

    def test_infer_in_channels(self):
        """Test infer_in_channels."""
        in_channels = infer_in_channels(self.dataset_config, self.cliq_lift_transform)
        assert in_channels == [7]
        
    def test_infer_num_cell_dimensions(self):
        """Test infer_num_cell_dimensions."""
        out = infere_num_cell_dimensions(None, [7, 7, 7])
        assert out == 3

        out = infere_num_cell_dimensions([1, 2, 3], [7, 7])
        assert out == 3
        
    def test_get_default_metrics(self):
        """Test get_default_metrics."""
        out = get_default_metrics("classification")
        assert out == ["accuracy", "precision", "recall", "auroc"]

        out = get_default_metrics("regression")
        assert out == ["mse", "mae"]

        with pytest.raises(ValueError, match="Invalid task") as e:
            get_default_metrics("some_task")
