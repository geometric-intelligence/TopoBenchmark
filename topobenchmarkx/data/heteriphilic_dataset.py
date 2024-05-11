import os.path as osp
from collections.abc import Callable
from typing import Optional, ClassVar

import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs

from topobenchmarkx.io.load.heterophilic import download_hetero_datasets, load_heterophilic_data
from topobenchmarkx.io.load.split_utils import random_splitting


class HeteroDataset(InMemoryDataset):
    r"""
    Dataset class for US County Demographics dataset.

    Args:
        root (str): Root directory where the dataset will be saved.
        name (str): Name of the dataset.
        parameters (DictConfig): Configuration parameters for the dataset.
        transform (Optional[Callable]): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed version.
            The transform function is applied to the loaded data before saving it.
        pre_transform (Optional[Callable]): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed version.
            The pre_transform function is applied to the data before the transform
            function is applied.
        pre_filter (Optional[Callable]): A function that takes in an
            `torch_geometric.data.Data` object and returns a boolean value
            indicating whether the data object should be included in the dataset.
        force_reload (bool): If set to True, the dataset will be re-downloaded
            even if it already exists on disk. (default: True)
        use_node_attr (bool): If set to True, the node attributes will be included
            in the dataset. (default: False)
        use_edge_attr (bool): If set to True, the edge attributes will be included
            in the dataset. (default: False)

    Attributes:
        URLS (dict): Dictionary containing the URLs for downloading the dataset.
        FILE_FORMAT (dict): Dictionary containing the file formats for the dataset.
        RAW_FILE_NAMES (dict): Dictionary containing the raw file names for the dataset.

    """

    RAW_FILE_NAMES: ClassVar = {}

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = True,
        use_node_attr: bool = False,
        use_edge_attr: bool = False,
    ) -> None:
        self.name = name #.replace("_", "-")
        self.parameters = parameters
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )

        # Step 3:Load the processed data
        # After the data has been downloaded from source
        # Then preprocessed to obtain x,y and saved into processed folder
        # We can now load the processed data from processed folder

        # Load the processed data
        data, _, _ = fs.torch_load(self.processed_paths[0])

        # Map the loaded data into
        data = Data.from_dict(data)

        # Step 5: Create the splits and upload desired fold
        splits = random_splitting(data.y, parameters=self.parameters)

        # Assign train val test masks to the graph
        data.train_mask = torch.from_numpy(splits["train"])
        data.val_mask = torch.from_numpy(splits["valid"])
        data.test_mask = torch.from_numpy(splits["test"])

        # Assign data object to self.data, to make it be prodessed by Dataset class
        self.data, self.slices = self.collate([data])

    # Do not forget to take care of properties
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
        """Spefify the downloaded raw fine name"""
        return [f"{self.name}.npz"]

    def download(self) -> None:
        """
        Downloads the dataset from the specified URL and saves it to the raw directory.

        Raises:
            FileNotFoundError: If the dataset URL is not found.
        """

        # Step 1: Download data from the source
        download_hetero_datasets(name=self.name, path=self.raw_dir)

    def process(self) -> None:
        """
        Process the data for the dataset.

        This method loads the US county demographics data, applies any pre-processing transformations if specified,
        and saves the processed data to the appropriate location.

        Returns:
            None
        """
    
        data = load_heterophilic_data(name=self.name, path=self.raw_dir)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.name}()"
