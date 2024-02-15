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
from topobenchmarkx.io.load.utils import (
    ensure_serializable,
    get_Planetoid_pyg,
    get_TUDataset_pyg,
    load_cell_complex_dataset,
    load_hypergraph_pickle_dataset,
    load_simplicial_dataset,
    load_split,
    make_hash,
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
    def __init__(self, parameters: DictConfig, transforms=None):
        super().__init__(parameters)
        self.parameters = parameters
        # Still not instantiated
        self.transforms_config = transforms

    def load(self):
        # Use self.transform_parameters to define unique save/load path for each transform parameters
        if self.transforms_config is None:
            transform_parameters = {"transform1": "Identity"}
            pre_transforms = None
            repo_name = "Identity"
        else:
            pre_transforms = hydra.utils.instantiate(self.transforms_config)

            transform_parameters = pre_transforms.parameters
            repo_name = pre_transforms.repo_name

        # Prepare the data directory name
        params_hash = make_hash(transform_parameters)
        data_dir = os.path.join(
            os.path.join(self.parameters["data_dir"], repo_name),
            f"{params_hash}",
        )

        if (
            self.parameters.data_name in ["Cora", "CiteSeer", "PubMed"]
            and self.parameters.data_type == "cocitation"
        ):
            dataset = torch_geometric.datasets.Planetoid(
                root=data_dir,  # self.parameters["data_dir"],
                name=self.parameters["data_name"],
                pre_transform=pre_transforms,
            )
            data = dataset.data

            data = load_split(data, self.parameters)
            dataset = CustomDataset([data])

        elif self.parameters.data_name in ["MUTAG", "ENZYMES", "PROTEINS", "COLLAB"]:
            dataset = torch_geometric.datasets.TUDataset(
                root=data_dir,  # self.parameters["data_dir"],
                name=self.parameters["data_name"],
                pre_transform=pre_transforms,
            )

            labels = dataset.y
            split_idx = rand_train_test_idx(labels)

            data_train_lst, data_val_lst, data_test_lst = [], [], []
            for i in range(len(dataset)):
                graph = dataset[i]

                if i in split_idx["train"]:
                    graph.train_mask = torch.Tensor([1]).long()
                    graph.val_mask = torch.Tensor([0]).long()
                    graph.test_mask = torch.Tensor([0]).long()
                    data_train_lst.append(graph)
                elif i in split_idx["valid"]:
                    graph.train_mask = torch.Tensor([0]).long()
                    graph.val_mask = torch.Tensor([1]).long()
                    graph.test_mask = torch.Tensor([0]).long()
                    data_val_lst.append(graph)
                elif i in split_idx["test"]:
                    graph.train_mask = torch.Tensor([0]).long()
                    graph.val_mask = torch.Tensor([0]).long()
                    graph.test_mask = torch.Tensor([1]).long()
                    data_test_lst.append(graph)
                else:
                    raise ValueError("Graph not in any split")

            # data_lst = [dataset[i] for i in range(len(dataset))]
            # REWRITE LATER

            dataset = [
                CustomDataset(data_train_lst),
                CustomDataset(data_val_lst),
                CustomDataset(data_test_lst),
            ]

        else:
            raise NotImplementedError(
                f"Dataset {self.parameters.data_name} not implemented"
            )

        # Check if root/params_dict.json exists, if not, save it
        path_transform_parameters = os.path.join(
            data_dir, "path_transform_parameters_dict.json"
        )
        transform_parameters = ensure_serializable(transform_parameters)
        if not os.path.exists(path_transform_parameters):
            with open(path_transform_parameters, "w") as f:
                json.dump(transform_parameters, f)
        else:
            # If path_transform_parameters exists, check if the transform_parameters are the same
            with open(path_transform_parameters, "r") as f:
                saved_transform_parameters = json.load(f)

            if saved_transform_parameters != transform_parameters:
                raise ValueError("Different transform parameters for the same data_dir")
            else:
                print(
                    f"Transform parameters are the same, using existing data_dir: {data_dir}"
                )

        return dataset


def rand_train_test_idx(
    label, train_prop=0.5, valid_prop=0.25, ignore_negative=True, balance=False, seed=0
):
    """Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not balance:
        if ignore_negative:
            labeled_nodes = torch.where(label != -1)[0]
        else:
            labeled_nodes = label

        n = labeled_nodes.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.as_tensor(np.random.permutation(n))

        train_indices = perm[:train_num]
        val_indices = perm[train_num : train_num + valid_num]
        test_indices = perm[train_num + valid_num :]

        if not ignore_negative:
            return train_indices, val_indices, test_indices

        train_idx = labeled_nodes[train_indices]
        valid_idx = labeled_nodes[val_indices]
        test_idx = labeled_nodes[test_indices]

        split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    else:
        #         ipdb.set_trace()
        indices = []
        for i in range(label.max() + 1):
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = int(train_prop / (label.max() + 1) * len(label))
        val_lb = int(valid_prop * len(label))
        train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]
        split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}

    # Save splits to disk

    return split_idx
