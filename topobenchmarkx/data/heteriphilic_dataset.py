import os
import os.path as osp
import urllib.request
from collections.abc import Callable
from typing import ClassVar

import numpy as np
import torch
import torch_geometric
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset

from topobenchmarkx.data.utils.split_utils import random_splitting


class HeteroDataset(InMemoryDataset):
    r"""Dataset class for heterophilic datasets.

    Args:
        root (str): Root directory where the dataset will be saved.
        name (str): Name of the dataset.
        parameters (DictConfig): Configuration parameters for the dataset.
        **kwargs: Additional keyword arguments.

    Attributes:
        RAW_FILE_NAMES (dict): Dictionary containing the raw file names for the dataset.
    """

    RAW_FILE_NAMES: ClassVar = {}

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
        **kwargs,
    ) -> None:
        self.name = name
        self.parameters = parameters
        super().__init__(
            root,
            **kwargs,
        )

        # Load the processed data
        data, _, _ = torch.load(self.processed_paths[0])

        # Map the loaded data into
        data = Data.from_dict(data)

        # Create the splits and upload desired fold
        splits = random_splitting(data.y, parameters=self.parameters)

        # Assign train val test masks to the graph
        data.train_mask = torch.from_numpy(splits["train"])
        data.val_mask = torch.from_numpy(splits["valid"])
        data.test_mask = torch.from_numpy(splits["test"])

        # Assign data object to self.data, to make it be prodessed by Dataset class
        self.data, self.slices = self.collate([data])

    def __repr__(self) -> str:
        return f"{self.name}(self.root={self.root}, self.name={self.name}, self.parameters={self.parameters}, self.transform={self.transform}, self.pre_transform={self.pre_transform}, self.pre_filter={self.pre_filter}, self.force_reload={self.force_reload})"
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    @property
    def raw_file_names(self) -> list[str]:
        return [f"{self.name}.npz"]

    def download(self) -> None:
        r"""Download a heterophilic dataset from the OpenGSL repository.
        
        Raises:
            Exception: If the download fails.
        """
        url = "https://github.com/OpenGSL/HeterophilousDatasets/raw/main/data/"
        name = f"{self.name}.npz"
        try:
            print(f"Downloading {name}")
            path2save = os.path.join(self.raw_dir, name)
            urllib.request.urlretrieve(url + name, path2save)
            print("Done!")
        except Exception as e:
            raise Exception(
                """Download failed! Make sure you have stable Internet connection and enter the right name"""
            ) from e

    def load_heterophilic_data(self, name, path):
        r"""Load a heterophilic dataset from a .npz file.
        
        Args:
            name (str): The name of the dataset.
            path (str): The path to the directory containing the dataset file.
        Returns:
            torch_geometric.data.Data: The dataset.
        """
        file_name = f"{self.name}.npz"

        data = np.load(os.path.join(self.raw_dir, file_name))

        x = torch.tensor(data["node_features"])
        y = torch.tensor(data["node_labels"])
        edge_index = torch.tensor(data["edges"]).T

        # Make edge_index undirected
        edge_index = torch_geometric.utils.to_undirected(edge_index)

        # Remove self-loops
        edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)

        data = torch_geometric.data.Data(x=x, y=y, edge_index=edge_index)
        return data

    def process(self) -> None:
        r"""Process the data for the dataset.

        This method loads the heterophilic data, applies any pre-processing transformations if specified,
        and saves the processed data to the appropriate location.
        """
        data = self.load_heterophilic_data(name=self.name, path=self.raw_dir)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])
