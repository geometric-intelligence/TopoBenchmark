import pytest
from unittest.mock import patch, MagicMock, Mock, PropertyMock, ANY
import os
import json
import torch_geometric
from omegaconf import DictConfig
from topobenchmarkx.dataloader import DataloadDataset
from topobenchmarkx.data.preprocess.preprocessor import PreProcessor
from ..._utils.flow_mocker import FlowMocker


@pytest.mark.usefixtures("mocker_fixture")
class TestPreProcessor:

    @pytest.fixture(autouse=True)
    def setup_method(self, mocker):
        
        # Setup test parameters
        self.dataset = MagicMock(spec=torch_geometric.data.Dataset)
        self.data_dir = "/fake/path"
        self.transforms_config = DictConfig({"transform": {"transform_name": "CellCycleLifting"}})

        params = [
            {"mock_inmemory_init": "torch_geometric.data.InMemoryDataset.__init__"},
            {"mock_save_transform": (PreProcessor, "save_transform_parameters")},
            {"mock_load": (PreProcessor, "load")},
            {"mock_len": (PreProcessor, "__len__"), "init_args": {"return_value":3}},
            {"mock_getitem": (PreProcessor, "get"), "init_args": {"return_value": "0"}},
        ]
        self.flow_mocker = FlowMocker(mocker, params)

        # Initialize PreProcessor
        self.preprocessor = PreProcessor(self.dataset, self.data_dir, None)
         

    def teardown_method(self):
        del self.preprocessor
        del self.flow_mocker
    
    def test_init(self):
        self.flow_mocker.get("mock_inmemory_init").assert_called_once_with(self.data_dir, None, None)
        self.flow_mocker.get("mock_load").assert_called_once_with(self.data_dir + "/processed/data.pt")
        assert self.preprocessor.transforms_applied == False
        assert self.preprocessor.data_list == ["0", "0", "0"]

    def test_init_with_transform(self, mocker):
        val_processed_paths = ["/some/path"]
        params = [
            {
                "assert_args": ("created_property", "processed_data_dir")
            },
            {
                "mock_inmemory_init": "torch_geometric.data.InMemoryDataset.__init__", 
                "assert_args": ("called_once_with", ANY, None, ANY)
            },
            {
                "mock_processed_paths": (PreProcessor, "processed_paths"),
                "init_args": {"property_val": val_processed_paths},
            },
            {
                "mock_save_transform": (PreProcessor, "save_transform_parameters"),
                "assert_args": ("created_property", "processed_paths")
            },
            {
                "mock_load": (PreProcessor, "load"),
                "assert_args": ("called_once_with", val_processed_paths[0])
            },
            {"mock_len": (PreProcessor, "__len__")},
            {"mock_getitem": (PreProcessor, "get")},
        ]
        self.flow_mocker = FlowMocker(mocker, params)
        self.preprocessor_with_tranform = PreProcessor(self.dataset, self.data_dir,  self.transforms_config)
        self.flow_mocker.assert_all(self.preprocessor_with_tranform)

        
    @patch('topobenchmarkx.data.preprocess.preprocessor.load_inductive_splits')
    def test_load_dataset_splits_inductive(self, mock_load_inductive_splits):
        split_params = DictConfig({"learning_setting": "inductive"})
        self.preprocessor.load_dataset_splits(split_params)
        mock_load_inductive_splits.assert_called_once_with(self.preprocessor, split_params)

    
    @patch('topobenchmarkx.data.preprocess.preprocessor.load_transductive_splits')
    def test_load_dataset_splits_transductive(self, mock_load_transductive_splits):
        split_params = DictConfig({"learning_setting": "transductive"})
        self.preprocessor.load_dataset_splits(split_params)
        mock_load_transductive_splits.assert_called_once_with(self.preprocessor, split_params)

    def test_invalid_learning_setting(self):
        split_params = DictConfig({"learning_setting": "invalid"})
        with pytest.raises(ValueError):
            self.preprocessor.load_dataset_splits(split_params)
    