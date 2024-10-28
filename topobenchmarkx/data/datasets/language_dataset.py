"""Dataset class for Language dataset. (see Tutorial "add_new_dataset.ipynb")"""

import os
import os.path as osp
import shutil
from typing import ClassVar
import json
import torch

from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import fs

from topobenchmarkx.data.utils import (
    download_file_from_drive,
    read_us_county_demos,
)


class LanguageDataset(InMemoryDataset):
    r"""Dataset class for Language dataset.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str
        Name of the dataset.
    parameters : DictConfig
        Configuration parameters for the dataset.

    Attributes
    ----------
    URLS (dict): Dictionary containing the URLs for downloading the dataset.
    FILE_FORMAT (dict): Dictionary containing the file formats for the dataset.
    RAW_FILE_NAMES (dict): Dictionary containing the raw file names for the dataset.
    """

    URLS: ClassVar = {
        "LanguageDataset": "https://drive.google.com/file/d/1jU8HGeXbMDIFph-kNsUmMwWHcCz44MC-/view?usp=sharing" 
    }

    FILE_FORMAT: ClassVar = {
        "LanguageDataset": "zip",
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
        # self.year = parameters.year
        # self.task_variable = parameters.task_variable
        super().__init__(
            root,
            force_reload=True,
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
        return f"{self.name}(self.root={self.root}, self.name={self.name}, self.force_reload={self.force_reload})"

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """
        self.processed_root = osp.join(
            self.root,
            self.name,
        )
        return osp.join(self.processed_root, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        return []

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name for the dataset.

        Returns
        -------
        str
            Processed file name.
        """
        return "data.pt"

    def download(self) -> None:
        r"""Download the dataset from a URL and saves it to the raw directory.

        Raises:
            FileNotFoundError: If the dataset URL is not found.
        """
        # Step 1: download data from the source
        self.url = self.URLS[self.name]
        self.file_format = self.FILE_FORMAT[self.name]
        download_file_from_drive(
            file_link=self.url,
            path_to_save=self.raw_dir,
            dataset_name=self.name,
            file_format=self.file_format,
        )

        # Step 2: extract file
        folder = self.raw_dir
        filename = f"{self.name}.{self.file_format}"
        path = osp.join(folder, filename)
        extract_zip(path, folder)
        os.unlink(path) # Delete zip file
        
        # Step 3: move the extracted files to the folder with corresponding name 
        # Move files from osp.join(folder, name_download) to folder
        for file in os.listdir(osp.join(folder, self.name)):
            if file.endswith('ipynb'):
                continue
            shutil.move(osp.join(folder, self.name, file), osp.join(folder, file))
        
        # Delete osp.join(folder, self.name) dir
        shutil.rmtree(osp.join(folder, self.name))

    def process(self) -> None:
        r"""Handle the data for the dataset.

        This method loads the Language dataser, applies any pre-
        processing transformations, and saves the processed data
        to the appropriate location.
        """
        # Step 1: extract the data
        folder = self.raw_dir
        with open(folder + '/token_tag_id_data.json', 'r') as file:
            token_tag_id_data = json.load(file)
        model_state_dict = torch.load(folder + '/Test_attention_all_head.pth', map_location=torch.device('cpu'))
        graph_sentences = []
        for sentence in range(len(model_state_dict)):
            ids = token_tag_id_data[str(sentence)]['ids']
            tokens = token_tag_id_data[str(sentence)]['tokens']
            tags = token_tag_id_data[str(sentence)]['tags']
            for head in range(len(model_state_dict[sentence])):
                attention_scores = model_state_dict[sentence][head]
                edge_index = []
                edge_attr = []
                for i in range(len(attention_scores)):
                    for j in range(len(attention_scores[0])):
                        edge_index.append([i,j])
                edge_index = torch.transpose(torch.FloatTensor(edge_index), 0, 1)
                # edge_attr = torch.transpose(torch.FloatTensor(edge_attr), 0, 1)
                # graph = Data(x=x, edge_index=edge_index, edge_attr = edge_attr)
                graph = Data(edge_index=edge_index, attention_scores = attention_scores.flatten(), 
                             attention_shape = attention_scores.shape, ids = ids, tokens = tokens, tags = tags, num_nodes=len(tokens))
                graph_sentences.append(graph)
        
        # Step 2: collate the graphs
        self.data, self.slices = self.collate(graph_sentences)
        self._data_list = None  # Reset cache.
        
        # Step 3: save processed data
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )

        self.graph = graph_sentences