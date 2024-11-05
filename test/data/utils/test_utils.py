"""Unit tests for data utils."""

import pytest
import torch
from topobenchmarkx.data.utils import *
import toponetx as tnx
from toponetx.classes import CellComplex
from topobenchmarkx.data.utils import get_complex_connectivity, select_neighborhoods_of_interest

class TestDataUtils:
    """Test data utils functions."""

    def setup_method(self):
        """Setup method."""
        self.complex = CellComplex()
        self.complex.add_cell([1, 2, 3],rank=2)
        self.complex.add_cell([3, 4, 5],rank=2)
        self.complex.add_cell([5, 6, 7],rank=2)
        self.neighborhoods1 = ['up_adjacency-0','2-up_adjacency-0','2-down_laplacian-2','2-down_adjacency-2','2-up_incidence-0','2-down_incidence-2']
        self.neighborhoods2 = ['incidence_0', 'down_laplacian_0', 'up_laplacian_0', 'adjacency_0', 'hodge_laplacian_1']
        
        
    def test_get_complex_connectivity(self):
        """Test get_complex_connectivity."""
        out = get_complex_connectivity(self.complex, 2, neighborhoods=self.neighborhoods2)
        assert ['incidence_0', 'down_laplacian_0', 'up_laplacian_0', 'adjacency_0'] in out.keys()
        
    def test_select_neighborhoods_of_interest(self):
        """Test select_neighborhoods_of_interest."""
        connectivity = get_complex_connectivity(self.complex, 2)
        out = select_neighborhoods_of_interest(connectivity, self.neighborhoods1)
        assert out == ['up_adjacency-0','2-up_adjacency-0','2-down_laplacian-2','2-down_adjacency-2','2-up_incidence-0','2-down_incidence-2']
        
        with pytest.raises(ValueError) as e:
            select_neighborhoods_of_interest(connectivity, ['invalid_neighborhood'])
        

    
    # def test_get_metric_value(self):
    #     """Test get_metric_value."""
    #     out = get_metric_value(self.metric_dict, "accuracy")
    #     assert out == 90.

    #     with pytest.raises(Exception) as e:
    #         get_metric_value(self.metric_dict, "some_metric")  
    
    # def test_extras(self):
    #     """Test extras."""
    #     extras(self.extras_cfg)
        
    
    # def test_task_wrapper(self):
    #     """Test task_wrapper."""
    #     d = DictConfig({'paths': {'output_dir': 'logs/'}})

    #     def task_func(cfg: DictConfig):
    #         """Task function for testing task_wrapper.
            
    #         Parameters
    #         ----------
    #         cfg : DictConfig
    #             A DictConfig object containing the config tree.

    #         Returns
    #         -------
    #         dict[str, Any], dict[str, Any]
    #             The metric and object dictionaries.
    #         """
    #         return {'accuracy': torch.tensor([90])}, {'model': 'model'}
        
        
    #     out = task_wrapper(task_func)(d)
        
    #     assert out[0]['accuracy'] == 90., "Metric dictionary not returned correctly."
    #     assert out[1]['model'] == 'model', "Object dictionary not returned correctly."
        
        
