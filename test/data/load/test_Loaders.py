"""Test the Cell and Simplicial Loaders."""

import pytest
from omegaconf import OmegaConf
from topobenchmarkx.data.loaders import CellComplexLoader, SimplicialLoader
import os
import hydra


class TestLoaders:
   """Test the CellComplexLoader and SimplicialLoader classes."""
   
   def setup_method(self):
       """Setup the test."""
       hydra.core.global_hydra.GlobalHydra.instance().clear()
       base_dir = os.path.abspath(__file__)
       # Go up four levels
       for _ in range(4):
           base_dir = os.path.dirname(base_dir)
       
       # Set up paths for both cell and simplicial configs
       self.cell_config_dir = os.path.join(base_dir, "configs/dataset/cell")
       self.simplicial_config_dir = os.path.join(base_dir, "configs/dataset/simplicial")

       # Get config files for both types
       self.cell_configs = [
           f for f in os.listdir(self.cell_config_dir) 
           if f.endswith(".yaml")
       ]
       self.simplicial_configs = [
           f for f in os.listdir(self.simplicial_config_dir) 
           if f.endswith(".yaml")
       ]

       # Exclude files if needed
       exclude_datasets = ["mantra_orintation.yaml", "mantra_name.yaml"]  # Add any datasets to exclude

       self.cell_configs = [
           f for f in self.cell_configs 
           if f not in exclude_datasets
       ]
       self.simplicial_configs = [
           f for f in self.simplicial_configs 
           if f not in exclude_datasets
       ]

       self.relative_config_dir = "../../../configs"

   def test_cell_loader(self):
       """Test the initialization and loading of the CellComplexLoader class."""
       for config_file in self.cell_configs:    
           with hydra.initialize(
               version_base="1.3",
               config_path=self.relative_config_dir,
               job_name="run"
           ):
               parameters = hydra.compose(
                   config_name="run.yaml",
                   overrides=[f"dataset=cell/{config_file}"], 
                   return_hydra_config=True
               )
               
           loader = CellComplexLoader(parameters.dataset.loader.parameters)
           dataset = loader.load()
           
           # Check basic properties
           assert dataset is not None
           #assert hasattr(dataset, "data_dir")
           
           # Test string representation
           repr(loader)

   def test_simplicial_loader(self):
       """Test the initialization and loading of the SimplicialLoader class."""
       for config_file in self.simplicial_configs:
            with hydra.initialize(
                version_base="1.3",
                config_path=self.relative_config_dir,
                job_name="run"
            ):
                parameters = hydra.compose(
                    config_name="run.yaml",
                    overrides=[f"dataset=simplicial/{config_file}"], 
                    return_hydra_config=True
                )
            
            loader = SimplicialLoader(parameters.dataset.loader.parameters)
            dataset = loader.load()
            
            # Check basic properties
            assert dataset is not None
            #assert hasattr(dataset, "data_dir")
            
            # Test string representation
            repr(loader)

   
   def test_missing_data_dir(self):
       """Test loaders with missing data directory."""
       config = OmegaConf.create({
           "data_dir": "/nonexistent/path",
           "data_name": "test"
       })
       
       with pytest.raises(Exception):  # Adjust exception type as needed
           loader = CellComplexLoader(config)
           loader.load()
       
       with pytest.raises(Exception):  # Adjust exception type as needed
           loader = SimplicialLoader(config)
           loader.load()

   def test_config_parameters(self):
       """Test loaders with different configuration parameters."""
       sample_config = OmegaConf.create({
           "data_dir": "test_dir",
           "data_name": "test_dataset",
           "additional_param": "test_value"
       })
       
       # Test CellComplexLoader
       cell_loader = CellComplexLoader(sample_config)
       assert cell_loader.parameters.data_dir == "test_dir"
       assert cell_loader.parameters.data_name == "test_dataset"
       
       # Test SimplicialLoader
       simplicial_loader = SimplicialLoader(sample_config)
       assert simplicial_loader.parameters.data_dir == "test_dir"
       assert simplicial_loader.parameters.data_name == "test_dataset"