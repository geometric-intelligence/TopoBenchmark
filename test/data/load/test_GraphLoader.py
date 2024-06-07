"""Test the GraphLoader class."""

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig

from topobenchmarkx.data.loaders import GraphLoader


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
            {"data_dir": "/path/to/data", "data_name": "UnknownDataset"}
        )
        loader = GraphLoader(parameters)
        with pytest.raises(NotImplementedError):
            loader.load()
