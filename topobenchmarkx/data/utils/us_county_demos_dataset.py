import os
import os.path as osp
import shutil
from typing import ClassVar

import numpy as np
import pandas as pd
import torch
import torch_geometric
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset, extract_zip

from topobenchmarkx.data.utils.download_utils import download_file_from_drive
from topobenchmarkx.data.utils.split_utils import random_splitting


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
    ) -> None:
        force_reload = parameters.get("force_reload", True)
        super().__init__(
            root,
            force_reload=force_reload,
        )
        self.name = name.replace("_", "-")
        self.parameters = parameters

        # Load the processed data
        data, _ = torch.load(self.processed_paths[0])
    
        # Map the loaded data into
        data = Data.from_dict(data) if isinstance(data, dict) else data

        # Create the splits and upload desired fold
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
        
        # Make sure the dataset will be reloaded during next run
        shutil.rmtree(self.raw_dir)
        # Get parent dir of self.processed_paths[0]
        processed_dir = os.path.abspath(os.path.join(self.processed_paths[0], os.pardir))
        shutil.rmtree(processed_dir)
        
    def __repr__(self) -> str:
        return f"{self.name}(self.root={self.root}, self.name={self.name}, self.parameters={self.parameters}, self.force_reload={self.force_reload})"

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        #names = ["county", f"{self.parameters.year}"]
        return ["county_graph.csv", f"county_stats_{self.parameters.year}.csv"]

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

        folder = self.raw_dir
        filename = f"{self.name}.{self.file_format}"
        path = osp.join(folder, filename)
        extract_zip(path, folder)
        # Delete zip file
        os.unlink(path)
        #shutil.rmtree(path)
        # Move files from osp.join(folder, self.name) to folder
        for file in os.listdir(osp.join(folder, self.name)):
            shutil.move(osp.join(folder, self.name, file), folder)
        
        # Delete osp.join(folder, self.name) dir
        shutil.rmtree(osp.join(folder, self.name))
        

    def process(self) -> None:
        r"""Process the data for the dataset.

        This method loads the US county demographics data, applies any pre-processing transformations if specified,
        and saves the processed data to the appropriate location.
        """
        data = self.load()

        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])
        
    def load(self):
        r"""Load US County Demos dataset considering the year and task_variable defined.
        
        Returns:
            torch_geometric.data.Data: Data object of the graph for the US County Demos dataset.
        """
        path = self.raw_dir
        year = self.parameters.year
        y_col = self.parameters.task_variable
        
        edges_df = pd.read_csv(f"{path}/county_graph.csv")
        stat = pd.read_csv(
            f"{path}/county_stats_{year}.csv", encoding="ISO-8859-1"
        )

        keep_cols = [
            "FIPS",
            "DEM",
            "GOP",
            "MedianIncome",
            "MigraRate",
            "BirthRate",
            "DeathRate",
            "BachelorRate",
            "UnemploymentRate",
        ]

        # Select columns, replace ',' with '.' and convert to numeric
        stat = stat.loc[:, keep_cols]
        stat["MedianIncome"] = stat["MedianIncome"].replace(",", ".", regex=True)
        stat = stat.apply(pd.to_numeric, errors="coerce")

        # Step 2: Substitute NaN values with column mean
        for column in stat.columns:
            if column != "FIPS":
                mean_value = stat[column].mean()
                stat[column].fillna(mean_value, inplace=True)
        stat = stat[keep_cols].dropna()

        # Delete edges that are not present in stat df
        unique_fips = stat["FIPS"].unique()

        src_ = edges_df["SRC"].apply(lambda x: x in unique_fips)
        dst_ = edges_df["DST"].apply(lambda x: x in unique_fips)

        edges_df = edges_df[src_ & dst_]

        # Remove rows from stat df where edges_df['SRC'] or edges_df['DST'] are not present
        stat = stat[
            stat["FIPS"].isin(edges_df["SRC"]) & stat["FIPS"].isin(edges_df["DST"])
        ]
        stat = stat.reset_index(drop=True)

        # Remove rows where SRC == DST
        edges_df = edges_df[edges_df["SRC"] != edges_df["DST"]]

        # Get torch_geometric edge_index format
        edge_index = torch.tensor(
            np.stack([edges_df["SRC"].to_numpy(), edges_df["DST"].to_numpy()])
        )

        # Make edge_index undirected
        edge_index = torch_geometric.utils.to_undirected(edge_index)

        # Convert edge_index back to pandas DataFrame
        edges_df = pd.DataFrame(edge_index.numpy().T, columns=["SRC", "DST"])

        del edge_index

        # Map stat['FIPS'].unique() to [0, ..., num_nodes]
        fips_map = {fips: i for i, fips in enumerate(stat["FIPS"].unique())}
        stat["FIPS"] = stat["FIPS"].map(fips_map)

        # Map edges_df['SRC'] and edges_df['DST'] to [0, ..., num_nodes]
        edges_df["SRC"] = edges_df["SRC"].map(fips_map)
        edges_df["DST"] = edges_df["DST"].map(fips_map)

        # Get torch_geometric edge_index format
        edge_index = torch.tensor(
            np.stack([edges_df["SRC"].to_numpy(), edges_df["DST"].to_numpy()])
        )

        # Remove isolated nodes (Note: this function maps the nodes to [0, ..., num_nodes] automatically)
        edge_index, _, mask = torch_geometric.utils.remove_isolated_nodes(
            edge_index
        )

        # Conver mask to index
        index = np.arange(mask.size(0))[mask]
        stat = stat.iloc[index]
        stat = stat.reset_index(drop=True)

        # Get new values for FIPS from current index
        # To understand why please print stat.iloc[[516, 517, 518, 519, 520]] for 2012 year
        # Basically the FIPS values has been shifted
        stat["FIPS"] = stat.reset_index()["index"]

        # Create Election variable
        stat["Election"] = (stat["DEM"] - stat["GOP"]) / (
            stat["DEM"] + stat["GOP"]
        )

        # Drop DEM and GOP columns and FIPS
        stat = stat.drop(columns=["DEM", "GOP", "FIPS"])

        # Prediction col
        x_col = list(stat.columns)
        x_col.remove(y_col)

        x = torch.tensor(stat[x_col].to_numpy(), dtype=torch.float32)
        y = torch.tensor(stat[y_col].to_numpy(), dtype=torch.float32)

        data = torch_geometric.data.Data(x=x, y=y, edge_index=edge_index)

        return data
