"""Comprehensive test suite for all dataset loaders."""
import os
import pytest
import torch
import hydra
from pathlib import Path
from typing import List, Tuple, Dict, Any
from omegaconf import DictConfig

class TestLoaders:
    """Comprehensive test suite for all dataset loaders."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment before each test method."""
        # Existing setup code remains the same
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        base_dir = Path(__file__).resolve().parents[3]
        self.config_files = self._gather_config_files(base_dir)
        self.relative_config_dir = "../../../configs"
        self.test_splits = ['train', 'val', 'test']

    # Existing helper methods remain the same
    def _gather_config_files(self, base_dir: Path) -> List[str]:
        """Gather all relevant config files.
        
        Parameters
        ----------
        base_dir : Path
            Base directory to start searching for config files.

        Returns
        -------
        List[str]
          List of config file paths.
        """
        config_files = []
        config_base_dir = base_dir / "configs/dataset"
        exclude_datasets = {"manual_dataset.yaml", "karate_club.yaml",
                            # Below the datasets that have some default transforms with we manually overriten with no_transform,
                            # due to lack of default transform for domain2domain
                            "REDDIT-BINARY.yaml", "IMDB-MULTI.yaml", "IMDB-BINARY.yaml", "ZINC.yaml"
                            }

        
        for dir_path in config_base_dir.iterdir():
            curr_dir = str(dir_path).split('/')[-1]
            if dir_path.is_dir():
                config_files.extend([
                    (curr_dir, f.name) for f in dir_path.glob("*.yaml")
                    if f.name not in exclude_datasets
                ])
        return config_files

    def _load_dataset(self, data_domain: str, config_file: str) -> Tuple[Any, Dict]:
        """Load dataset with given config file.

        Parameters
        ----------
        data_domain : str
            Name of the data domain.
        config_file : str
          Name of the config file.
        
        Returns
        -------
        Tuple[Any, Dict]
          Tuple containing the dataset and dataset directory.
        """
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="run"
        ):
            print('Current config file: ', config_file)
            parameters = hydra.compose(
                config_name="run.yaml",
                overrides=[f"dataset={data_domain}/{config_file}"], 
                return_hydra_config=True
                
            )
            dataset_loader = hydra.utils.instantiate(parameters.dataset.loader)
            return dataset_loader.load()

    def test_dataset_loading_states(self):
        """Test different states and scenarios during dataset loading."""
        for config_data in self.config_files:
            data_domain, config_file = config_data
            dataset, _ = self._load_dataset(data_domain, config_file)
            
            # Test dataset size and dimensions
            if hasattr(dataset, "data"):
                assert dataset.data.x.size(0) > 0, "Empty node features"
                assert dataset.data.y.size(0) > 0, "Empty labels"
            else: 
                assert dataset[0].x.size(0) > 0, "Empty node features"
                assert dataset[0].y.size(0) > 0, "Empty labels"
            
            # Test node feature dimensions
            if hasattr(dataset, 'num_node_features'):
                assert dataset.data.x.size(1) == dataset.num_node_features
            
            # Test label dimensions
            if hasattr(dataset, 'num_classes'):
                assert torch.max(dataset.data.y) < dataset.num_classes

            repr(dataset)

    
                


    # # New test methods for improved coverage

    # def test_validate_parameters(self):
    #     """Test parameter validation for dataset loaders."""
    #     for config_data in self.config_files:
    #         data_domain, config_file = config_data
    #         with hydra.initialize(
    #             version_base="1.3",
    #             config_path=self.relative_config_dir,
    #             job_name="run"
    #         ):
    #             parameters = hydra.compose(
    #                 config_name="run.yaml",
    #                 overrides=[f"dataset={data_domain}/{config_file}", "transforms=no_transform"], 
    #                 return_hydra_config=True
    #             )
                
    #             # Test invalid dataset name
    #             invalid_params = parameters.dataset.loader.copy()
    #             invalid_params.parameters.data_name = "invalid_dataset"
    #             with pytest.raises(ValueError, match="not supported"):
    #                 hydra.utils.instantiate(invalid_params)
                
    #             # Test invalid dataset type
    #             invalid_type_params = parameters.dataset.loader.copy()
    #             invalid_type_params.parameters.data_type = "invalid_type"
    #             with pytest.raises(ValueError, match="not supported"):
    #                 hydra.utils.instantiate(invalid_type_params)

    # def test_data_directory_handling(self):
    #     """Test data directory path handling and validation."""
    #     for config_file in self.config_files:
    #         with hydra.initialize(
    #             version_base="1.3",
    #             config_path=self.relative_config_dir,
    #             job_name="run"
    #         ):
    #             parameters = hydra.compose(
    #                 config_name="run.yaml",
    #                 overrides=[f"dataset=$graph/{config_file}"], 
    #                 return_hydra_config=True
    #             )
                
    #             loader = hydra.utils.instantiate(parameters.dataset.loader)
                
    #             # Test get_data_dir method
    #             data_dir = loader.get_data_dir()
    #             assert isinstance(data_dir, Path)
    #             assert data_dir.exists()
                
    #             # Test root data directory exists
    #             assert loader.root_data_dir.exists()


    # def test_dataset_initialization_edge_cases(self):
    #     """Test edge cases in dataset initialization."""
    #     for config_file in self.config_files:
    #         with hydra.initialize(
    #             version_base="1.3",
    #             config_path=self.relative_config_dir,
    #             job_name="run"
    #         ):
    #             parameters = hydra.compose(
    #                 config_name="run.yaml",
    #                 overrides=["dataset=${loader.parameters.data_domain}/"+f"{config_file}"], 
    #                 return_hydra_config=True
    #             )
                
    #             # Test initialization with minimum required parameters
    #             min_params = DictConfig({
    #                 "data_dir": parameters.dataset.loader.data_dir,
    #                 "data_name": parameters.dataset.loader.data_name,
    #                 "data_type": parameters.dataset.loader.data_type
    #             })
    #             loader = hydra.utils.instantiate(parameters.dataset.loader.__class__, **min_params)
    #             assert loader is not None

    # def test_dataset_specific_attributes(self):
    #     """Test attributes specific to different dataset types."""
    #     for config_file in self.config_files:
    #         dataset, _ = self._load_dataset(config_file)
            
    #         # Test HeterophilousGraph specific attributes
    #         if hasattr(dataset, 'data_type') and dataset.data_type == "heterophilous":
    #             assert hasattr(dataset.data, 'edge_index')
    #             assert isinstance(dataset.data.edge_index, torch.Tensor)
    #             assert dataset.data.edge_index.dim() == 2
    #             assert dataset.data.edge_index.size(0) == 2
            
    #         # Test USCountyDemos specific attributes
    #         if hasattr(dataset, 'data_type') and "county-demos" in str(dataset.data_type):
    #             assert hasattr(dataset, 'task_variable')
    #             if hasattr(dataset.data, 'edge_attr'):
    #                 assert isinstance(dataset.data.edge_attr, torch.Tensor)

    # def test_data_consistency(self):
    #     """Test data consistency across multiple loads."""
    #     for config_file in self.config_files:
    #         # Load dataset twice
    #         dataset1, _ = self._load_dataset(config_file)
    #         dataset2, _ = self._load_dataset(config_file)
            
    #         # Compare data attributes
    #         assert torch.equal(dataset1.data.x, dataset2.data.x)
    #         assert torch.equal(dataset1.data.y, dataset2.data.y)
    #         if hasattr(dataset1.data, 'edge_index'):
    #             assert torch.equal(dataset1.data.edge_index, dataset2.data.edge_index)
