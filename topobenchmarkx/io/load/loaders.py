# import copy
import json
import os

import hydra
import numpy as np
import toponetx.datasets.graph as graph
import torch
import torch_geometric
from omegaconf import DictConfig

from topobenchmarkx.data.datasets import CustomDataset
from topobenchmarkx.io.load.loader import AbstractLoader
from topobenchmarkx.io.load.preprocessor import Preprocessor
from topobenchmarkx.io.load.utils import (
    load_cell_complex_dataset,
    load_graph_cocitation_split,
    load_graph_tudataset_split,
    load_hypergraph_pickle_dataset,
    load_simplicial_dataset,
    load_split,
)


class CellComplexLoader(AbstractLoader):
    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.parameters = parameters

    def load(
        self,
    ):
        data = load_cell_complex_dataset(self.parameters)
        dataset = CustomDataset([data])
        return dataset


class SimplicialLoader(AbstractLoader):
    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.parameters = parameters

    def load(
        self,
    ):
        data = load_simplicial_dataset(self.parameters)
        dataset = CustomDataset([data])
        return dataset


class HypergraphLoader(AbstractLoader):
    def __init__(self, parameters: DictConfig, transforms=None):
        super().__init__(parameters)
        self.parameters = parameters
        self.transforms_config = transforms

    def load(
        self,
    ):
        data = load_hypergraph_pickle_dataset(self.parameters)
        data = load_split(data, self.parameters)
        dataset = CustomDataset([data])

        # pre_transforms_dict = hydra.utils.instantiate(self.transforms_config)

        # pre_transforms_dict = hydra.utils.instantiate(self.transforms_config)
        # pre_transforms = torch_geometric.transforms.Compose(
        #     list(pre_transforms_dict.values())
        # )
        # repo_name = "_".join(list(self.transforms_config.keys()))
        # transform_parameters = {
        #     transform_name: transform.parameters
        #     for transform_name, transform in pre_transforms_dict.items()
        # }
        # # Prepare the data directory name
        # params_hash = make_hash(transform_parameters)
        # data_dir = os.path.join(
        #     os.path.join(
        #         os.path.join(self.parameters["data_dir"], self.parameters["data_name"]),
        #         repo_name,
        #     ),
        #     f"{params_hash}",
        # )

        # if pre_transforms is not None:
        #     dataset = PreprocessedDataset(data_dir, dataset, pre_transforms)
        # We need to add checks that:
        # All nodes belong to some edge, in case some not, create selfedge

        return dataset


class GraphLoader(AbstractLoader):
    def __init__(self, parameters: DictConfig, transforms=None):
        super().__init__(parameters)
        self.parameters = parameters
        # Still not instantiated
        self.transforms_config = transforms

    def load(self):
        data_dir = os.path.join(
            self.parameters["data_dir"], self.parameters["data_name"]
        )
        if (
            self.parameters.data_name.lower() in ["cora", "citeseer", "pubmed"]
            and self.parameters.data_type == "cocitation"
        ):
            dataset = torch_geometric.datasets.Planetoid(
                root=self.parameters["data_dir"],
                name=self.parameters["data_name"],
            )
            if self.transforms_config is not None:
                dataset = Preprocessor(data_dir, dataset, self.transforms_config)
            dataset = load_graph_cocitation_split(dataset, data_dir, self.parameters)

        elif self.parameters.data_name in ["MUTAG", "ENZYMES", "PROTEINS", "COLLAB"]:
            dataset = torch_geometric.datasets.TUDataset(
                root=self.parameters["data_dir"],
                name=self.parameters["data_name"],
                use_node_attr=False,
            )
            if self.transforms_config is not None:
                dataset = Preprocessor(data_dir, dataset, self.transforms_config)
            dataset = load_graph_tudataset_split(dataset, data_dir, self.parameters)

        else:
            raise NotImplementedError(
                f"Dataset {self.parameters.data_name} not implemented"
            )

        return dataset
