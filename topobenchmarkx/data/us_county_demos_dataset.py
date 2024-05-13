import os.path as osp
from collections.abc import Callable
from typing import ClassVar

import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs

from topobenchmarkx.io.load.download_utils import download_file_from_drive
from topobenchmarkx.io.load.split_utils import random_splitting
from topobenchmarkx.io.load.us_county_demos import load_us_county_demos


class USCountyDemosDataset(InMemoryDataset):
    r"""Dataset class for US County Demographics dataset.

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

    URLS: ClassVar = {
        # 'contact-high-school': 'https://drive.google.com/open?id=1VA2P62awVYgluOIh1W4NZQQgkQCBk-Eu',
        "US-county-demos": "https://drive.google.com/file/d/1FNF_LbByhYNICPNdT6tMaJI9FxuSvvLK/view?usp=sharing",
    }

    FILE_FORMAT: ClassVar = {
        # 'contact-high-school': 'tar.gz',
        "US-county-demos": "zip",
    }

    RAW_FILE_NAMES: ClassVar = {}

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        force_reload: bool = True,
        use_node_attr: bool = False,
        use_edge_attr: bool = False,
    ) -> None:
        self.name = name.replace("_", "-")
        self.parameters = parameters
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            force_reload=force_reload,
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

        # Standardize the node features respecting train mask
        data.x = (data.x - data.x[data.train_mask].mean(0)) / data.x[
            data.train_mask
        ].std(0)
        data.y = (data.y - data.y[data.train_mask].mean(0)) / data.y[
            data.train_mask
        ].std(0)

        # Assign data object to self.data, to make it be prodessed by Dataset class
        self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        names = ["", f"_{self.parameters.year}"]
        return [f"{self.name}_{name}.txt" for name in names]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self) -> None:
        """Downloads the dataset from the specified URL and saves it to the raw
        directory.

        Raises:
            FileNotFoundError: If the dataset URL is not found.
        """

        # Step 1: Download data from the source
        self.url = self.URLS[self.name]
        self.file_format = self.FILE_FORMAT[self.name]

        download_file_from_drive(
            file_link=self.url,
            path_to_save=self.raw_dir,
            dataset_name=self.name,
            file_format=self.file_format,
        )

        # Extract the downloaded file if it is compressed
        fs.cp(
            f"{self.raw_dir}/{self.name}.{self.file_format}",
            self.raw_dir,
            extract=True,
        )

        # Move the etracted files to the datasets/domain/dataset_name/raw/ directory
        for filename in fs.ls(osp.join(self.raw_dir, self.name)):
            fs.mv(filename, osp.join(self.raw_dir, osp.basename(filename)))
        fs.rm(osp.join(self.raw_dir, self.name))

        # Delete also f'{self.raw_dir}/{self.name}.{self.file_format}'
        fs.rm(f"{self.raw_dir}/{self.name}.{self.file_format}")

    def process(self) -> None:
        """Process the data for the dataset.

        This method loads the US county demographics data, applies any pre-processing transformations if specified,
        and saves the processed data to the appropriate location.

        Returns:
            None
        """
        data = load_us_county_demos(
            self.raw_dir,
            year=self.parameters.year,
            y_col=self.parameters.task_variable,
        )

        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.name}()"
