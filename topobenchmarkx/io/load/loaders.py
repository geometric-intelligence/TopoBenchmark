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
    get_tran_val_test_graph_datasets,
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

            dataset = load_graph_cocitation_split(dataset, self.parameters)

        elif self.parameters.data_name in [
            "MUTAG",
            "ENZYMES",
            "PROTEINS",
            "COLLAB",
            "IMDB-BINARY",
            "IMDB-MULTI",
            "REDDIT-BINARY",
            "NCI1",
            "NCI109",
        ]:
            dataset = torch_geometric.datasets.TUDataset(
                root=self.parameters["data_dir"],
                name=self.parameters["data_name"],
                use_node_attr=False,
            )
            if self.transforms_config is not None:
                dataset = Preprocessor(data_dir, dataset, self.transforms_config)
            dataset = load_graph_tudataset_split(dataset, self.parameters)

        elif self.parameters.data_name in ["ZINC"]:
            datasets = []
            for split in ["train", "val", "test"]:
                datasets.append(
                    torch_geometric.datasets.ZINC(
                        root=self.parameters["data_dir"],
                        subset=True,
                        split=split,
                    )
                )

            assert self.parameters.split_type == "fixed"
            # The splits are predefined
            # Extract and prepare split_idx
            split_idx = {"train": np.arange(len(datasets[0]))}

            split_idx["valid"] = np.arange(
                len(datasets[0]), len(datasets[0]) + len(datasets[1])
            )

            split_idx["test"] = np.arange(
                len(datasets[0]) + len(datasets[1]),
                len(datasets[0]) + len(datasets[1]) + len(datasets[2]),
            )

            # Join dataset to process it
            joined_dataset = datasets[0] + datasets[1] + datasets[2]

            if self.transforms_config is not None:
                joined_dataset = Preprocessor(
                    data_dir,
                    joined_dataset,
                    self.transforms_config,
                )

            # Split back the into train/val/test datasets
            dataset = get_tran_val_test_graph_datasets(joined_dataset, split_idx)

        elif self.parameters.data_name in ["AQSOL"]:
            datasets = []
            for split in ["train", "val", "test"]:
                datasets.append(
                    torch_geometric.datasets.AQSOL(
                        root=self.parameters["data_dir"],
                        split=split,
                    )
                )
            # The splits are predefined
            # Extract and prepare split_idx
            split_idx = {"train": np.arange(len(datasets[0]))}

            split_idx["valid"] = np.arange(
                len(datasets[0]), len(datasets[0]) + len(datasets[1])
            )

            split_idx["test"] = np.arange(
                len(datasets[0]) + len(datasets[1]),
                len(datasets[0]) + len(datasets[1]) + len(datasets[2]),
            )

            # Join dataset to process it
            joined_dataset = datasets[0] + datasets[1] + datasets[2]

            if self.transforms_config is not None:
                joined_dataset = Preprocessor(
                    data_dir,
                    joined_dataset,
                    self.transforms_config,
                )

            # Split back the into train/val/test datasets
            dataset = get_tran_val_test_graph_datasets(joined_dataset, split_idx)
        else:
            raise NotImplementedError(
                f"Dataset {self.parameters.data_name} not implemented"
            )

        return dataset
    

class ManualGraphLoader(AbstractLoader):
    def __init__(self, parameters: DictConfig, transforms=None):
        super().__init__(parameters)
        self.parameters = parameters
        # Still not instantiated
        self.transforms_config = transforms

    def load(self):
        import networkx as nx
        # Define the vertices (just 7 vertices)
        vertices = [i for i in range(9)]
        y = [0, 1, 1, 1, 0, 0, 0, 0, 0]
        # Define the edges
        edges = [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [1, 3],
            [2, 3],
            [5, 2],
            [5, 6],
            [6, 3],
            [2, 6],
            [5, 7],
            [2,8],
            [0,8],
        ]

        # Define the tetrahedrons
        tetrahedrons = [[0, 1, 2, 3], [0, 1, 2, 4]]

        # Add tetrahedrons
        for tetrahedron in tetrahedrons:
            for i in range(len(tetrahedron)):
                for j in range(i + 1, len(tetrahedron)):
                    edges.append([tetrahedron[i], tetrahedron[j]])

        # Create a graph
        G = nx.Graph()

        # Add vertices
        G.add_nodes_from(vertices)

        # Add edges
        G.add_edges_from(edges)
        G.to_undirected()
        edge_list = torch.Tensor(list(G.edges())).T.long()
        #edge_list = torch.sparse_coo_tensor(edge_list, torch.ones(edge_list.shape[1]), (len(vertices), len(vertices)))
        data = torch_geometric.data.Data(x=torch.ones((len(vertices), 1)).float(), edge_index=edge_list, num_nodes=len(vertices), y=torch.tensor(y))
        
        if self.transforms_config is not None:
            data_dir = os.path.join(
            self.parameters["data_dir"], self.parameters["data_name"])
            processor_dataset = Preprocessor(data_dir, data, self.transforms_config)
        
        dataset = CustomDataset([processor_dataset[0]])
        return dataset
