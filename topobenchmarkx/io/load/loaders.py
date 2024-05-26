# import copy
import os

import networkx as nx
import numpy as np
import torch
import torch_geometric
from omegaconf import DictConfig

from topobenchmarkx.dataset.datasets import CustomDataset
from topobenchmarkx.dataset.heteriphilic_dataset import HeteroDataset
from topobenchmarkx.dataset.utils.us_county_demos_dataset import USCountyDemosDataset
from topobenchmarkx.io.load.loader import AbstractLoader
from topobenchmarkx.io.load.preprocessor import Preprocessor
from topobenchmarkx.dataset.utils.split_utils import (
    assing_train_val_test_mask_to_graphs,
    load_graph_cocitation_split,
    load_graph_tudataset_split,
    load_hypergraph_coauthorship_split,
)
from topobenchmarkx.io.load.utils import (
    load_cell_complex_dataset,
    load_hypergraph_pickle_dataset,
    load_simplicial_dataset,
)


class CellComplexLoader(AbstractLoader):
    r"""Loader for cell complex datasets.

    Args:
        parameters (DictConfig): Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.parameters = parameters
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameters={self.parameters})"

    def load(
        self,
    ) -> CustomDataset:
        r"""Load cell complex dataset.

        Returns:
            CustomDataset: CustomDataset object containing the loaded data.
        """
        data = load_cell_complex_dataset(self.parameters)
        dataset = CustomDataset([data])
        return dataset


class SimplicialLoader(AbstractLoader):
    r"""Loader for simplicial datasets.

    Args:
        parameters (DictConfig): Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.parameters = parameters
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameters={self.parameters})"

    def load(
        self,
    ) -> CustomDataset:
        r"""Load simplicial dataset.

        Returns:
            CustomDataset: CustomDataset object containing the loaded data.
        """
        data = load_simplicial_dataset(self.parameters)
        dataset = CustomDataset([data])
        return dataset


class HypergraphLoader(AbstractLoader):
    r"""Loader for hypergraph datasets.

    Args:
        parameters (DictConfig): Configuration parameters.
        transforms (DictConfig, optional): The parameters for the transforms to be applied to the dataset. (default: None)
    """

    def __init__(self, parameters: DictConfig, transforms=None):
        super().__init__(parameters)
        self.parameters = parameters
        self.transforms_config = transforms
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameters={self.parameters}, transforms={self.transforms_config})"

    def load(
        self,
    ) -> CustomDataset:
        r"""Load hypergraph dataset.
        
        Returns:
            CustomDataset: CustomDataset object containing the loaded data.
        """
        data = load_hypergraph_pickle_dataset(self.parameters)
        data = load_hypergraph_coauthorship_split(data, self.parameters)
        dataset = CustomDataset([data])
        return dataset


class GraphLoader(AbstractLoader):
    r"""Loader for graph datasets.

    Args:
        parameters (DictConfig): Configuration parameters. The parameters must contain the following keys:
            - data_dir (str): The directory where the dataset is stored.
            - data_name (str): The name of the dataset.
            - data_type (str): The type of the dataset.
            - split_type (str): The type of split to be used. It can be "fixed", "random", or "k-fold".
            
            If split_type is "random", the parameters must also contain the following keys:
                - data_seed (int): The seed for the split.
                - data_split_dir (str): The directory where the split is stored.
                - train_prop (float): The proportion of the training set.
            If split_type is "k-fold", the parameters must also contain the following keys:
                - data_split_dir (str): The directory where the split is stored.
                - k (int): The number of folds.
                - data_seed (int): The seed for the split.
            The parameters can be defined in a yaml file and then loaded using `omegaconf.OmegaConf.load('path/to/dataset/config.yaml')`.
        transforms (DictConfig, optional): The parameters for the transforms to be applied to the dataset. The parameters for a transformation can be defined in a yaml file and then loaded using `omegaconf.OmegaConf.load('path/to/transform/config.yaml'). (default: None)
    """
    def __init__(self, parameters: DictConfig, transforms=None):
        super().__init__(parameters)
        self.parameters = parameters
        # Still not instantiated
        self.transforms_config = transforms
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameters={self.parameters}, transforms={self.transforms_config})"

    def load(self) -> CustomDataset:
        r"""Load graph dataset.

        Returns:
            CustomDataset: CustomDataset object containing the loaded data.
        """
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
                dataset = Preprocessor(
                    data_dir, dataset, self.transforms_config
                )

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
                dataset = Preprocessor(
                    data_dir, dataset, self.transforms_config
                )
            dataset = load_graph_tudataset_split(dataset, self.parameters)

        elif self.parameters.data_name in ["ZINC"]:
            datasets = [
                torch_geometric.datasets.ZINC(
                    root=self.parameters["data_dir"],
                    subset=True,
                    split=split,
                )
                for split in ["train", "val", "test"]
            ]

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
            dataset = assing_train_val_test_mask_to_graphs(
                joined_dataset, split_idx
            )

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
            dataset = assing_train_val_test_mask_to_graphs(
                joined_dataset, split_idx
            )

        elif self.parameters.data_name in ["US-county-demos"]:
            dataset = USCountyDemosDataset(
                root=self.parameters["data_dir"],
                name=self.parameters["data_name"],
                parameters=self.parameters,
            )

            if self.transforms_config is not None:
                # force_reload=True because in this datasets many variables can be trated as y
                dataset = Preprocessor(
                    data_dir,
                    dataset,
                    self.transforms_config,
                    force_reload=True,
                )

            # We need to map original dataset into custom one to make batching work
            dataset = CustomDataset([dataset[0]])

        elif self.parameters.data_name in [
            "amazon_ratings",
            "questions",
            "minesweeper",
            "roman_empire",
            "tolokers",
        ]:
            dataset = HeteroDataset(
                root=self.parameters["data_dir"],
                name=self.parameters["data_name"],
                parameters=self.parameters,
            )

            if self.transforms_config is not None:
                # force_reload=True because in this datasets many variables can be trated as y
                dataset = Preprocessor(
                    data_dir,
                    dataset,
                    self.transforms_config,
                    force_reload=False,
                )

            # We need to map original dataset into custom one to make batching work
            dataset = CustomDataset([dataset[0]])

        else:
            raise NotImplementedError(
                f"Dataset {self.parameters.data_name} not implemented"
            )

        return dataset


class ManualGraphLoader(AbstractLoader):
    r"""Loader for manual graph datasets.

    Args:
        parameters (DictConfig): Configuration parameters.
        transforms (DictConfig, optional): The parameters for the transforms to be applied to the dataset. (default: None)
    """

    def __init__(self, parameters: DictConfig, transforms=None):
        super().__init__(parameters)
        self.parameters = parameters
        # Still not instantiated
        self.transforms_config = transforms
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameters={self.parameters}, transforms={self.transforms_config})"

    def load(self) -> CustomDataset:
        r"""Load manual graph dataset.

        Returns:
            CustomDataset: CustomDataset object containing the loaded data.
        """
        data = manual_graph()

        if self.transforms_config is not None:
            data_dir = os.path.join(
                self.parameters["data_dir"], self.parameters["data_name"]
            )
            processor_dataset = Preprocessor(
                data_dir, data, self.transforms_config
            )

        dataset = CustomDataset([processor_dataset[0]])
        return dataset


def manual_graph():
    """Create a manual graph for testing purposes."""
    # Define the vertices (just 9 vertices)
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
        [2, 8],
        [0, 8],
    ]

    # Define the tetrahedrons
    tetrahedrons = [[0, 1, 2, 3], [0, 1, 2, 4]]

    # Add tetrahedrons
    for tetrahedron in tetrahedrons:
        for i in range(len(tetrahedron)):
            for j in range(i + 1, len(tetrahedron)):
                edges.append([tetrahedron[i], tetrahedron[j]])  # noqa: PERF401

    # Create a graph
    G = nx.Graph()

    # Add vertices
    G.add_nodes_from(vertices)

    # Add edges
    G.add_edges_from(edges)
    G.to_undirected()
    edge_list = torch.Tensor(list(G.edges())).T.long()

    # Generate feature from 0 to 9
    x = (
        torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000, 10000])
        .unsqueeze(1)
        .float()
    )

    data = torch_geometric.data.Data(
        x=x,
        edge_index=edge_list,
        num_nodes=len(vertices),
        y=torch.tensor(y),
    )
    return data


def manual_simple_graph():
    """Create a manual graph for testing purposes."""
    # Define the vertices (just 8 vertices)
    vertices = [i for i in range(8)]
    y = [0, 1, 1, 1, 0, 0, 0, 0]
    # Define the edges
    edges = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 2],
        [2, 3],
        [5, 2],
        [5, 6],
        [6, 3],
        [5, 7],
        [2, 7],
        [0, 7],
    ]

    # Define the tetrahedrons
    tetrahedrons = [[0, 1, 2, 4]]

    # Add tetrahedrons
    for tetrahedron in tetrahedrons:
        for i in range(len(tetrahedron)):
            for j in range(i + 1, len(tetrahedron)):
                edges.append([tetrahedron[i], tetrahedron[j]])  # noqa: PERF401

    # Create a graph
    G = nx.Graph()

    # Add vertices
    G.add_nodes_from(vertices)

    # Add edges
    G.add_edges_from(edges)
    G.to_undirected()
    edge_list = torch.Tensor(list(G.edges())).T.long()

    # Generate feature from 0 to 9
    x = torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000]).unsqueeze(1).float()

    data = torch_geometric.data.Data(
        x=x,
        edge_index=edge_list,
        num_nodes=len(vertices),
        y=torch.tensor(y),
    )
    return data
