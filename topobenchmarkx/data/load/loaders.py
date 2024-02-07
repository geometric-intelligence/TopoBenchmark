import copy
import json
import os

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


def make_hash(o):
    """
    Makes a hash from a dictionary, list, tuple or set to any level, that contains
    only other hashable types (including any lists, tuples, sets, and
    dictionaries).
    """

    if isinstance(o, (set, tuple, list)):
        return tuple([make_hash(e) for e in o])

    elif not isinstance(o, dict):
        return hash(o)

    new_o = copy.deepcopy(o)
    for k, v in new_o.items():
        new_o[k] = make_hash(v)

    return hash(tuple(frozenset(sorted(new_o.items()))))


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


class GraphLoader(AbstractLoader):
    def __init__(self, parameters: DictConfig, transform_parameters=None):
        super().__init__(parameters)
        self.parameters = parameters
        self.transform_parameters = transform_parameters

    def load(self, transforms=None):
        # Use self.transform_parameters to define unique save/load path for each transform parameters
        params_hash = make_hash(self.transform_parameters)
        data_dir = os.path.join(self.parameters["data_dir"], f"{params_hash}")
        if (
            self.parameters.data_name in ["Cora", "CiteSeer", "PubMed"]
            and self.parameters.data_type == "cocitation"
        ):
            dataset = torch_geometric.datasets.Planetoid(
                root=data_dir,  # self.parameters["data_dir"],
                name=self.parameters["data_name"],
                pre_transform=transforms,
            )
            data = dataset.data
            data = load_split(data, self.parameters)
            dataset = CustomDataset([data])

        elif self.parameters.data_name in ["MUTAG", "ENZYMES", "PROTEINS", "COLLAB"]:
            dataset = torch_geometric.datasets.TUDataset(
                root=data_dir,  # self.parameters["data_dir"],
                name=self.parameters["data_name"],
                pre_transform=transforms,
            )
            data_lst = [dataset[i] for i in range(len(dataset))]
            dataset = CustomDataset(data_lst)

        else:
            raise NotImplementedError(
                f"Dataset {self.parameters .data_name} not implemented"
            )

        # Check if root/params_dict.json exists, if not, save it
        path_transform_parameters = os.path.join(
            data_dir, "path_transform_parameters_dict.json"
        )
        if not os.path.exists(path_transform_parameters):
            with open(path_transform_parameters, "w") as f:
                json.dump(self.transform_parameters, f)
        else:
            # If path_transform_parameters exists, check if the transform_parameters are the same
            with open(path_transform_parameters, "r") as f:
                saved_transform_parameters = json.load(f)

            if saved_transform_parameters != self.transform_parameters:
                raise ValueError("Different transform parameters for the same data_dir")
            else:
                print(
                    f"Transform parameters are the same, using existing data_dir: {data_dir}"
                )

        return dataset


# class PYGLoader(AbstractLoader):
#     def __init__(self, parameters: DictConfig):
#         super().__init__(parameters)
#         self.parameters = parameters

#     def load(self):
#         if (
#             self.parameters.data_name in ["Cora", "CiteSeer", "PubMed"]
#             and self.parameters.data_type == "cocitation"
#         ):
#             data = get_Planetoid_pyg(cfg=self.parameters)
#             data = load_split(data, self.parameters)
#             dataset = CustomDataset([data])

#         elif self.parameters.data_name == ["MUTAG", "ENZYMES", "PROTEINS", "COLLAB"]:
#             data_lst = get_TUDataset_pyg(cfg=self.parameters)
#             dataset = CustomDataset(data_lst)

#         else:
#             raise NotImplementedError(
#                 f"Dataset {self.parameters.data_name} not implemented"
#             )

#         return dataset
