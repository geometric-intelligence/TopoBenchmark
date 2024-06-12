"""Unit tests for config instantiators."""

import pytest
from omegaconf import OmegaConf, DictConfig
import torch
from topobenchmarkx.utils.utils import extras, get_metric_value, task_wrapper

class TestUtils:
    """Test config instantiators."""

    def setup_method(self):
        """Setup method."""
        self.metric_dict = {'accuracy': torch.tensor([90])}
        self.extras_cfg = OmegaConf.load("configs/extras/default.yaml")

    
    def test_get_metric_value(self):
        """Test get_metric_value."""
        out = get_metric_value(self.metric_dict, "accuracy")
        assert out == 90.

        with pytest.raises(Exception) as e:
            get_metric_value(self.metric_dict, "some_metric")  
    
    def test_extras(self):
        """Test extras."""
        extras(self.extras_cfg)
        
    
    def test_task_wrapper(self):
        """Test task_wrapper."""
        d = DictConfig({'paths': {'output_dir': 'logs/'}})

        def task_func(cfg: DictConfig):
            """Task function for testing task_wrapper.
            
            Parameters
            ----------
            cfg : DictConfig
                A DictConfig object containing the config tree.

            Returns
            -------
            dict[str, Any], dict[str, Any]
                The metric and object dictionaries.
            """
            return {'accuracy': torch.tensor([90])}, {'model': 'model'}
        
        
        out = task_wrapper(task_func)(d)
        
        assert out[0]['accuracy'] == 90., "Metric dictionary not returned correctly."
        assert out[1]['model'] == 'model', "Object dictionary not returned correctly."
        
        
