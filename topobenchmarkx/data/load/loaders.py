import os

import numpy as np
import rootutils
import torch_geometric
from omegaconf import DictConfig

from topobenchmarkx.data.load.base import AbstractLoader
from topobenchmarkx.data.utils.concat2geometric_dataset import (
    ConcatToGeometricDataset,
)
from topobenchmarkx.data.utils.datasets import CustomDataset
from topobenchmarkx.data.heteriphilic_dataset import HeteroDataset
from topobenchmarkx.data.us_county_demos_dataset import USCountyDemosDataset
from topobenchmarkx.data.utils.utils import (
    load_cell_complex_dataset,
    load_hypergraph_pickle_dataset,
    load_manual_graph,
    load_simplicial_dataset,
)


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
    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.parameters = parameters
        
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

    def load(self) -> torch_geometric.data.Dataset:
        r"""Load graph dataset.

        Parameters
        ----------
        None

        Returns
        -------
        torch_geometric.data.Dataset
            torch_geometric.data.Dataset object containing the loaded data.
        """
        # Define the path to the data directory
        root_folder = rootutils.find_root()
        root_data_dir = os.path.join(root_folder, self.parameters["data_dir"])

        self.data_dir = os.path.join(root_data_dir, self.parameters["data_name"])
        if (
            self.parameters.data_name.lower() in ["cora", "citeseer", "pubmed"]
            and self.parameters.data_type == "cocitation"
        ):
            dataset = torch_geometric.datasets.Planetoid(
                root=root_data_dir,
                name=self.parameters["data_name"],
            )

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
                root=root_data_dir,
                name=self.parameters["data_name"],
                use_node_attr=False,
            )

        elif self.parameters.data_name in ["ZINC", "AQSOL"]:
            datasets = []
            for split in ["train", "val", "test"]:
                if self.parameters.data_name == "ZINC":
                    datasets.append(
                        torch_geometric.datasets.ZINC(
                            root=root_data_dir,
                            subset=True,
                            split=split,
                        )
                    )
                elif self.parameters.data_name == "AQSOL":
                    datasets.append(
                        torch_geometric.datasets.AQSOL(
                            root=root_data_dir,
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
            dataset = datasets[0] + datasets[1] + datasets[2]
            dataset = ConcatToGeometricDataset(dataset)

        elif self.parameters.data_name in ["manual"]:
            data = load_manual_graph()
            dataset = CustomDataset([data], self.data_dir)

        else:
            raise NotImplementedError(
                f"Dataset {self.parameters.data_name} not implemented"
            )

        return dataset


class CellComplexLoader(AbstractLoader):
    r"""Loader for cell complex datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.parameters = parameters

    def load(
        self,
    ) -> torch_geometric.data.Dataset:
        r"""Load cell complex dataset.

        Parameters
        ----------
        None

        Returns
        -------
        torch_geometric.data.Dataset
            torch_geometric.data.Dataset object containing the loaded data.
        """
        return load_cell_complex_dataset(self.parameters)


class SimplicialLoader(AbstractLoader):
    r"""Loader for simplicial datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.parameters = parameters

    def load(
        self,
    ) -> torch_geometric.data.Dataset:
        r"""Load simplicial dataset.

        Parameters
        ----------
        None

        Returns
        -------
        torch_geometric.data.Dataset
            torch_geometric.data.Dataset object containing the loaded data.
        """
        return load_simplicial_dataset(self.parameters)


class HypergraphLoader(AbstractLoader):
    r"""Loader for hypergraph datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.parameters = parameters

    def load(
        self,
    ) -> torch_geometric.data.Dataset:
        r"""Load hypergraph dataset.

        Parameters
        ----------
        None

        Returns
        -------
        torch_geometric.data.Dataset
            torch_geometric.data.Dataset object containing the loaded data.
        """
        return load_hypergraph_pickle_dataset(self.parameters)
