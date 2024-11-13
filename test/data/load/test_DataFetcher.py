# """Test suite for all dataset loaders."""

# import os
# import pytest
# import torch
# import hydra
# from pathlib import Path
# from typing import List, Tuple, Dict, Any

# class TestLoaders:
#     """Comprehensive test suite for all dataset loaders."""
    
#     @pytest.fixture(autouse=True)
#     def setup(self):
#         """Setup test environment before each test method."""
#         # Clear any existing Hydra instance
#         hydra.core.global_hydra.GlobalHydra.instance().clear()
        
#         # Get project root directory
#         base_dir = Path(__file__).resolve().parents[3]
        
#         # Gather all config files
#         self.config_files = self._gather_config_files(base_dir)
#         self.relative_config_dir = "../../../configs"
        
#         # Set up test data
#         self.test_splits = ['train', 'val', 'test']
        
#     def _gather_config_files(self, base_dir: Path) -> List[str]:
#         """Gather all relevant config files."""
#         config_files = []
#         config_base_dir = base_dir / "configs/dataset"
        
#         # Excluded datasets
#         exclude_datasets = {"manual_dataset.yaml", "karate_club.yaml"}
        
#         for dir_path in config_base_dir.iterdir():
#             if dir_path.is_dir():
#                 config_files.extend([
#                     f.name for f in dir_path.glob("*.yaml")
#                     if f.name not in exclude_datasets
#                 ])
        
#         return config_files
    
#     def _load_dataset(self, config_file: str) -> Tuple[Any, Dict]:
#         """Load dataset with given config file."""
#         with hydra.initialize(
#             version_base="1.3",
#             config_path=self.relative_config_dir,
#             job_name="run"
#         ):
#             parameters = hydra.compose(
#                 config_name="run.yaml",
#                 return_hydra_config=True
#             )
            
#             dataset_loader = hydra.utils.instantiate(parameters.dataset.loader)
#             return dataset_loader.load()

#     def test_dataset_initialization(self):
#         """Test initialization of all dataset loaders."""
#         for config_file in self.config_files:
#             dataset, data_dir = self._load_dataset(config_file)
            
#             # Test basic dataset properties
#             assert dataset is not None, f"Dataset failed to load for {config_file}"
#             assert hasattr(dataset, 'data'), f"Dataset missing data attribute for {config_file}"
            
#             # Test dataset representation
#             assert repr(dataset), f"Dataset repr failed for {config_file}"
            
    

#     def test_dataset_attributes(self):
#         """Test dataset attributes and properties."""
#         for config_file in self.config_files:
#             dataset, _ = self._load_dataset(config_file)
            
#             # Test common attributes
#             assert hasattr(dataset, 'num_node_features'), f"Missing num_node_features for {config_file}"
#             assert hasattr(dataset, 'num_classes'), f"Missing num_classes for {config_file}"
            
#             # Test data object attributes
#             if hasattr(dataset, 'data'):
#                 assert hasattr(dataset.data, 'x'), f"Missing node features for {config_file}"
#                 assert hasattr(dataset.data, 'y'), f"Missing labels for {config_file}"
                
#                 # Verify data types
#                 assert isinstance(dataset.data.x, torch.Tensor), f"Node features not tensor for {config_file}"
#                 assert isinstance(dataset.data.y, torch.Tensor), f"Labels not tensor for {config_file}"

#     def test_dataset_processing(self):
#         """Test dataset processing methods."""
#         for config_file in self.config_files:
#             dataset, data_dir = self._load_dataset(config_file)
            
#             # Test data processing flags
#             if hasattr(dataset, 'processed_file_names'):
#                 assert isinstance(dataset.processed_file_names, list) or isinstance(dataset.processed_file_names, str), \
#                     f"Processed file names not list for {config_file}"
            
#             # Test raw file handling if applicable
#             if hasattr(dataset, 'raw_file_names'):
#                 assert isinstance(dataset.raw_file_names, list), \
#                     f"Raw file names not list for {config_file}"

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
        exclude_datasets = {"manual_dataset.yaml", "karate_club.yaml"}
        
        for dir_path in config_base_dir.iterdir():
            if dir_path.is_dir():
                config_files.extend([
                    f.name for f in dir_path.glob("*.yaml")
                    if f.name not in exclude_datasets
                ])
        return config_files

    def _load_dataset(self, config_file: str) -> Tuple[Any, Dict]:
        """Load dataset with given config file.

        Parameters
        ----------
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
            parameters = hydra.compose(
                config_name="run.yaml",
                return_hydra_config=True
            )
            dataset_loader = hydra.utils.instantiate(parameters.dataset.loader)
            return dataset_loader.load()

    # New test methods for improved coverage

    def test_validate_parameters(self):
        """Test parameter validation for dataset loaders."""
        for config_file in self.config_files:
            with hydra.initialize(
                version_base="1.3",
                config_path=self.relative_config_dir,
                job_name="run"
            ):
                parameters = hydra.compose(
                    config_name="run.yaml",
                    return_hydra_config=True
                )
                
                # Test invalid dataset name
                invalid_params = parameters.dataset.loader.copy()
                invalid_params.parameters.data_name = "invalid_dataset"
                with pytest.raises(ValueError, match="not supported"):
                    hydra.utils.instantiate(invalid_params)
                
                # Test invalid dataset type
                invalid_type_params = parameters.dataset.loader.copy()
                invalid_type_params.parameters.data_type = "invalid_type"
                with pytest.raises(ValueError, match="not supported"):
                    hydra.utils.instantiate(invalid_type_params)

    def test_data_directory_handling(self):
        """Test data directory path handling and validation."""
        for config_file in self.config_files:
            with hydra.initialize(
                version_base="1.3",
                config_path=self.relative_config_dir,
                job_name="run"
            ):
                parameters = hydra.compose(
                    config_name="run.yaml",
                    
                    overrides=[f"dataset=graph/{f}"],

                    return_hydra_config=True
                )
                
                loader = hydra.utils.instantiate(parameters.dataset.loader)
                
                # Test get_data_dir method
                data_dir = loader.get_data_dir()
                assert isinstance(data_dir, Path)
                assert data_dir.exists()
                
                # Test root data directory exists
                assert loader.root_data_dir.exists()

    def test_dataset_loading_states(self):
        """Test different states and scenarios during dataset loading."""
        for config_file in self.config_files:
            dataset, _ = self._load_dataset(config_file)
            
            # Test dataset size and dimensions
            assert dataset.data.x.size(0) > 0, "Empty node features"
            assert dataset.data.y.size(0) > 0, "Empty labels"
            
            # Test node feature dimensions
            if hasattr(dataset, 'num_node_features'):
                assert dataset.data.x.size(1) == dataset.num_node_features
            
            # Test label dimensions
            if hasattr(dataset, 'num_classes'):
                assert torch.max(dataset.data.y) < dataset.num_classes

    def test_dataset_initialization_edge_cases(self):
        """Test edge cases in dataset initialization."""
        for config_file in self.config_files:
            with hydra.initialize(
                version_base="1.3",
                config_path=self.relative_config_dir,
                job_name="run"
            ):
                parameters = hydra.compose(
                    config_name="run.yaml",
                    return_hydra_config=True
                )
                
                # Test initialization with minimum required parameters
                min_params = DictConfig({
                    "data_dir": parameters.dataset.loader.data_dir,
                    "data_name": parameters.dataset.loader.data_name,
                    "data_type": parameters.dataset.loader.data_type
                })
                loader = hydra.utils.instantiate(parameters.dataset.loader.__class__, **min_params)
                assert loader is not None

    def test_dataset_specific_attributes(self):
        """Test attributes specific to different dataset types."""
        for config_file in self.config_files:
            dataset, _ = self._load_dataset(config_file)
            
            # Test HeterophilousGraph specific attributes
            if hasattr(dataset, 'data_type') and dataset.data_type == "heterophilous":
                assert hasattr(dataset.data, 'edge_index')
                assert isinstance(dataset.data.edge_index, torch.Tensor)
                assert dataset.data.edge_index.dim() == 2
                assert dataset.data.edge_index.size(0) == 2
            
            # Test USCountyDemos specific attributes
            if hasattr(dataset, 'data_type') and "county-demos" in str(dataset.data_type):
                assert hasattr(dataset, 'task_variable')
                if hasattr(dataset.data, 'edge_attr'):
                    assert isinstance(dataset.data.edge_attr, torch.Tensor)

    def test_data_consistency(self):
        """Test data consistency across multiple loads."""
        for config_file in self.config_files:
            # Load dataset twice
            dataset1, _ = self._load_dataset(config_file)
            dataset2, _ = self._load_dataset(config_file)
            
            # Compare data attributes
            assert torch.equal(dataset1.data.x, dataset2.data.x)
            assert torch.equal(dataset1.data.y, dataset2.data.y)
            if hasattr(dataset1.data, 'edge_index'):
                assert torch.equal(dataset1.data.edge_index, dataset2.data.edge_index)
