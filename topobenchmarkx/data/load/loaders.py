import hydra
import torch
import torch_geometric
from omegaconf import DictConfig
from torch_geometric.data import Data

from topobenchmarkx.data.datasets import CustomDataset
from topobenchmarkx.data.load.loader import AbstractLoader
from topobenchmarkx.data.utils import (
    get_Planetoid_pyg,
    get_TUDataset_pyg,
    load_hypergraph_pickle_dataset,
    load_split,
)


class HypergraphLoader(AbstractLoader):
    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.parameters = parameters

    def load(
        self,
    ):
        data = load_hypergraph_pickle_dataset(self.parameters)
        data = load_split(data, self.parameters)
        dataset = CustomDataset([data])
        # We need to add checks that:
        # All nodes belong to some edge, in case some not, create selfedge

        return dataset


class PYGLoader(AbstractLoader):
    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)

    def load(self):
        if (
            self.parameters.data_name in ["Cora", "CiteSeer", "PubMed"]
            and self.parameters.data_type == "cocitation"
        ):
            data = get_Planetoid_pyg(cfg=self.parameters)
            data = load_split(data, self.parameters)
            dataset = CustomDataset([data])
        elif self.parameters.data_name == ["MUTAG", "ENZYMES", "PROTEINS", "COLLAB"]:
            data_lst = get_TUDataset_pyg(cfg=self.parameters)
            dataset = CustomDataset([data_lst])
        else:
            raise NotImplementedError(
                f"Dataset {self.parameters.data_name} not implemented"
            )

        return dataset


class GraphLoader(AbstractLoader):
    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)

    def load(self):
        transform = hydra.utils.instantiate(self.cfg["transform"])
        if (
            self.cfg.data_name in ["Cora", "CiteSeer", "PubMed"]
            and self.cfg.data_type == "cocitation"
        ):
            dataset = torch_geometric.datasets.Planetoid(
                root=self.cfg["data_dir"],
                name=self.cfg["data_name"],
                pre_transform=transform,
            )
        else:
            raise NotImplementedError(f"Dataset {self.cfg.data_name} not implemented")

        return dataset
