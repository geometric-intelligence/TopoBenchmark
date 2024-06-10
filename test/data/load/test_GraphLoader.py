"""Test the GraphLoader class."""

from unittest.mock import MagicMock, patch
import torch
import pytest
from omegaconf import DictConfig, OmegaConf

from topobenchmarkx.data.loaders import GraphLoader
from topobenchmarkx.data.utils.io_utils import (
    read_us_county_demos,
    load_hypergraph_pickle_dataset
)
from topobenchmarkx.data.utils.utils import load_simplicial_dataset, load_manual_graph, ensure_serializable

class TestGraphLoader:
    """Test the GraphLoader class."""
    
    def setup_method(self):
        """Setup the test."""
        self.parameters = DictConfig(
            {
                "data_dir": "/path/to/data",
                "data_name": "Cora",
                "data_type": "cocitation",
                "split_type": "fixed",
            }
        )

    def teardown_method(self):
        """Test teardown."""
        del self.parameters

    def test_init(self):
        """Test the initialization of the GraphLoader class."""
        loader = GraphLoader(self.parameters)
        assert loader.parameters == self.parameters

    @patch("torch_geometric.datasets.Planetoid")
    def test_load_planetoid(self, mock_planetoid):
        """Test loading a Planetoid dataset.
        
        Parameters
        ----------
        mock_planetoid : MagicMock
            A mock of the Planetoid class.
        """
        parameters = DictConfig(
            {
                "data_dir": "/path/to/data",
                "data_name": "Cora",
                "data_type": "cocitation",
                "split_type": "fixed",
            }
        )
        mock_planetoid.return_value = MagicMock()
        loader = GraphLoader(parameters)
        dataset, data_dir = loader.load()

        mock_planetoid.assert_called_once_with(
            root="/path/to/data", name="Cora"
        )
        assert data_dir == "/path/to/data/Cora"

    @patch("torch_geometric.datasets.TUDataset")
    def test_load_tu_dataset(self, mock_tudataset):
        """Test loading a TUDataset.
        
        Parameters
        ----------
        mock_tudataset : MagicMock
            A mock of the TUDataset class.
        """
        parameters = DictConfig(
            {"data_dir": "/path/to/data", "data_name": "MUTAG"}
        )
        mock_tudataset.return_value = MagicMock()
        loader = GraphLoader(parameters)
        dataset, data_dir = loader.load()

        mock_tudataset.assert_called_once_with(
            root="/path/to/data", name="MUTAG", use_node_attr=False
        )
        assert data_dir == "/path/to/data/MUTAG"

    @patch("torch_geometric.datasets.ZINC")
    @patch("torch_geometric.datasets.AQSOL")
    def test_load_fixed_splits(self, *mock_datasets):
        """Test loading datasets with fixed splits.
        
        Parameters
        ----------
        *mock_datasets : list[MagicMock]
            A list of mocks of the datasets.
        """
        # The cases must be in reverse order of @patch(...)
        cases = [
            ("AQSOL", dict()),
            ("ZINC", {"subset": True}),
        ]
        for i, mock_dataset in enumerate(mock_datasets):
            data_name = cases[i][0]
            data_kwargs = cases[i][1]
            parameters = DictConfig(
                {"data_dir": "/path/to/data", "data_name": data_name}
            )
            mock_dataset.return_value = MagicMock()
            loader = GraphLoader(parameters)
            dataset, data_dir = loader.load()

            for split in ["train", "val", "test"]:
                mock_dataset.assert_any_call(
                    root="/path/to/data", split=split, **data_kwargs
                )

            assert data_dir == "/path/to/data"

    @patch("torch_geometric.datasets.HeterophilousGraphDataset")
    def test_load_heterophilous(self, mock_dataset):
        """Test loading a HeterophilousGraphDataset.
        
        Parameters
        ----------
        mock_dataset : MagicMock
            A mock of the HeterophilousGraphDataset class.
        """
        parameters = DictConfig(
            {"data_dir": "/path/to/data", "data_name": "amazon_ratings"}
        )
        mock_dataset.return_value = MagicMock()
        loader = GraphLoader(parameters)
        dataset, data_dir = loader.load()

        mock_dataset.assert_called_once_with(
            root="/path/to/data", name="amazon_ratings"
        )
        assert data_dir == "/path/to/data/amazon_ratings"

    def test_load_unsupported_dataset(self):
        """Test loading an unsupported dataset."""
        parameters = DictConfig(
            {"data_dir": "path/to/data", "data_name": "UnknownDataset"}
        )
        loader = GraphLoader(parameters)
        with pytest.raises(NotImplementedError):
            loader.load()

    def test_load_us_county_dataset(self):
        """Test loading US County dataset."""
        loader_config = {
            "data_domain": "graph",
            "data_type": "cornel",
            "data_name": "US-county-demos",
            "year": 2012,
            "data_dir": "./.test_tmp/data/US_COUNTY",
            "task_variable": "MigraRate"
        }
        loader_config = OmegaConf.create(loader_config)
        graph_loader = GraphLoader(loader_config)
        dataset, dataset_dir = graph_loader.load()
        data = dataset[0]

        assert data.x.shape == (3224, 6)
        
        read_us_county_demos(loader_config.data_dir + "/US-county-demos/raw")

    def test_load_karate_dataset(self):
        """ Test loading the KarateClub dataset."""
        cfg = {
            "data_name": "KarateClub",
        }
        cfg = OmegaConf.create(cfg)
        data = load_simplicial_dataset(cfg)
        assert data.y.shape == (34,)
        
    def test_manual_loader(self):
        """ Test loading a manual graph."""
        data = load_manual_graph()
        assert data.y.shape == (8,)
        
    def test_ensure_serialization(self):
        """Test that the class is serializable."""
        obj1 = (1,2)
        obj2 = set([1,2,3,3])
        cfg = {
            "data_name": "KarateClub",
        }
        obj3 = OmegaConf.create(cfg)
        obj4 = load_manual_graph()
        
        res1 = ensure_serializable(obj1)
        res2 = ensure_serializable(obj2)
        res3 = ensure_serializable(obj3)
        res4 = ensure_serializable(obj4)
        
        assert res1 is not None
        assert res2 is not None
        assert res3 is not None
        assert res4 is None
        