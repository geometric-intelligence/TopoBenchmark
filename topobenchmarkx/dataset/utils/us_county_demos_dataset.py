import os
import os.path as osp
import shutil
from typing import ClassVar

from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import fs

from topobenchmarkx.dataset.utils import (
    download_file_from_drive,
    read_us_county_demos,
)


class USCountyDemosDataset(InMemoryDataset):
    r"""Dataset class for US County Demographics dataset.

    Args:
        root (str): Root directory where the dataset will be saved.
        name (str): Name of the dataset.
        parameters (DictConfig): Configuration parameters for the dataset.
        transform (Callable, optional): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed version.
            The transform function is applied to the loaded data before saving it. (default: None)
        pre_transform (Callable, optional): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed version.
            The pre_transform function is applied to the data before the transform
            function is applied. (default: None)
        pre_filter (Callable, optional): A function that takes in an
            `torch_geometric.data.Data` object and returns a boolean value
            indicating whether the data object should be included in the dataset. (default: None)
        force_reload (bool, optional): If set to True, the dataset will be re-downloaded
            even if it already exists on disk. (default: True)
        use_node_attr (bool, optional): If set to True, the node attributes will be included
            in the dataset. (default: False)
        use_edge_attr (bool, optional): If set to True, the edge attributes will be included
            in the dataset. (default: False)

    Attributes:
        URLS (dict): Dictionary containing the URLs for downloading the dataset.
        FILE_FORMAT (dict): Dictionary containing the file formats for the dataset.
        RAW_FILE_NAMES (dict): Dictionary containing the raw file names for the dataset.
    """

    URLS: ClassVar = {
        "US-county-demos": "https://drive.google.com/file/d/1FNF_LbByhYNICPNdT6tMaJI9FxuSvvLK/view?usp=sharing",
    }

    FILE_FORMAT: ClassVar = {
        "US-county-demos": "zip",
    }

    RAW_FILE_NAMES: ClassVar = {}

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
    ) -> None:
        self.name = name
        self.parameters = parameters
        self.year = parameters.year
        self.task_variable = parameters.task_variable
        super().__init__(
            root,
        )
        
        out = fs.torch_load(self.processed_paths[0])
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:  # Backward compatibility.
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        assert isinstance(self._data, Data)
        
    def __repr__(self) -> str:
        return f"{self.name}(self.root={self.root}, self.name={self.name}, self.parameters={self.parameters}, self.force_reload={self.force_reload})"

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        self.processed_root = osp.join(self.root, self.name, "_".join([str(self.year),self.task_variable]))
        return osp.join(self.processed_root, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        return ["county_graph.csv", f"county_stats_{self.year}.csv"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"
    
    def download(self) -> None:
        r"""Downloads the dataset from the specified URL and saves it to the raw
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
        # Extract zip file
        folder = self.raw_dir
        filename = f"{self.name}.{self.file_format}"
        path = osp.join(folder, filename)
        extract_zip(path, folder)
        # Delete zip file
        os.unlink(path)
        # Move files from osp.join(folder, name_download) to folder
        for file in os.listdir(osp.join(folder, self.name)):
            shutil.move(osp.join(folder, self.name, file), folder)
        # Delete osp.join(folder, self.name) dir
        shutil.rmtree(osp.join(folder, self.name))

    def process(self) -> None:
        r"""Process the data for the dataset.

        This method loads the US county demographics data, applies any pre-processing transformations if specified,
        and saves the processed data to the appropriate location.
        """
        data = read_us_county_demos(self.raw_dir, self.year, self.task_variable)
        data_list = [data]
        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache.
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )
