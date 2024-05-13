import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from topobenchmarkx.data.datasets import CustomDataset


# Generate splits in different fasions
def k_fold_split(labels, parameters):
    """Returns train and valid indices as in K-Fold Cross-Validation. If the
    split already exists it loads it automatically, otherwise it creates the
    split file for the subsequent runs.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor.
    parameters : DictConfig
        Configuration parameters.

    Returns
    -------
    dict
        Dictionary containing the train, validation and test indices.
    """

    data_dir = parameters.data_split_dir
    k = parameters.k
    fold = parameters.data_seed
    assert fold < k, "data_seed needs to be less than k"

    torch.manual_seed(0)
    np.random.seed(0)

    split_dir = os.path.join(data_dir, f"{k}-fold")

    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)

    split_path = os.path.join(split_dir, f"{fold}.npz")
    if not os.path.isfile(split_path):
        n = labels.shape[0]
        x_idx = np.arange(n)
        x_idx = np.random.permutation(x_idx)
        labels = labels[x_idx]

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        for fold_n, (train_idx, valid_idx) in enumerate(
            skf.split(x_idx, labels)
        ):
            split_idx = {
                "train": train_idx,
                "valid": valid_idx,
                "test": valid_idx,
            }

            # Check that all nodes/graph have been assigned to some split
            assert np.all(
                np.sort(
                    np.array(
                        split_idx["train"].tolist()
                        + split_idx["valid"].tolist()
                    )
                )
                == np.sort(np.arange(len(labels)))
            ), "Not every sample has been loaded."
            split_path = os.path.join(split_dir, f"{fold_n}.npz")

            np.savez(split_path, **split_idx)

    split_path = os.path.join(split_dir, f"{fold}.npz")
    split_idx = np.load(split_path)

    # Check that all nodes/graph have been assigned to some split
    assert (
        np.unique(
            np.array(
                split_idx["train"].tolist()
                + split_idx["valid"].tolist()
                + split_idx["test"].tolist()
            )
        ).shape[0]
        == labels.shape[0]
    ), "Not all nodes within splits"

    return split_idx


def random_splitting(labels, parameters, global_data_seed=42):
    """Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks
    randomly splits label into train/valid/test splits.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor.
    parameters : DictConfig
        Configuration parameters.
    global_data_seed : int
        Seed for the random number generator.

    Returns
    -------
    dict
        Dictionary containing the train, validation and test indices.
    """
    fold = parameters["data_seed"]
    data_dir = parameters["data_split_dir"]
    train_prop = parameters["train_prop"]
    valid_prop = (1 - train_prop) / 2

    # Create split directory if it does not exist
    split_dir = os.path.join(
        data_dir, f"train_prop={train_prop}_global_seed={global_data_seed}"
    )
    generate_splits = False
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)
        generate_splits = True

    # Generate splits if they do not exist
    if generate_splits:
        # Set initial seed
        torch.manual_seed(global_data_seed)
        np.random.seed(global_data_seed)
        # Generate a split
        n = labels.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        # Generate 10 splits
        for fold_n in range(10):
            # Permute indices
            perm = torch.as_tensor(np.random.permutation(n))

            train_indices = perm[:train_num]
            val_indices = perm[train_num : train_num + valid_num]
            test_indices = perm[train_num + valid_num :]
            split_idx = {
                "train": train_indices,
                "valid": val_indices,
                "test": test_indices,
            }

            # Save generated split
            split_path = os.path.join(split_dir, f"{fold_n}.npz")
            np.savez(split_path, **split_idx)

    # Load the split
    split_path = os.path.join(split_dir, f"{fold}.npz")
    split_idx = np.load(split_path)

    # Check that all nodes/graph have been assigned to some split
    assert (
        np.unique(
            np.array(
                split_idx["train"].tolist()
                + split_idx["valid"].tolist()
                + split_idx["test"].tolist()
            )
        ).shape[0]
        == labels.shape[0]
    ), "Not all nodes within splits"

    return split_idx


def load_split(data, cfg, train_prop=0.5):
    r"""Loads the split for generated by rand_train_test_idx function.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph dataset.
    cfg : DictConfig
        Configuration parameters.
    train_prop : float
        Proportion of training data.

    Returns
    -------
    torch_geometric.data.Data
        Graph dataset with the specified split.
    """

    data_dir = os.path.join(cfg["data_split_dir"], f"train_prop={train_prop}")
    load_path = f"{data_dir}/split_{cfg['data_seed']}.npz"
    splits = np.load(load_path, allow_pickle=True)

    # Upload masks
    data.train_mask = torch.from_numpy(splits["train"])
    data.val_mask = torch.from_numpy(splits["valid"])
    data.test_mask = torch.from_numpy(splits["test"])

    # Check that all nodes assigned to splits
    assert (
        torch.unique(
            torch.concat([data.train_mask, data.val_mask, data.test_mask])
        ).shape[0]
        == data.num_nodes
    ), "Not all nodes within splits"
    return data


def assing_train_val_test_mask_to_graphs(dataset, split_idx):
    r"""Splits the graph dataset into train, validation, and test datasets.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Graph dataset.
    split_idx : dict
        Dictionary containing the indices for the train, validation, and test splits.

    Returns
    -------
    datasets : list
        List containing the train, validation, and test datasets.
    """
    data_train_lst, data_val_lst, data_test_lst = [], [], []

    # Go over each of the graph and assign correct label
    for i in range(len(dataset)):
        graph = dataset[i]
        assigned = False
        if i in split_idx["train"]:
            graph.train_mask = torch.Tensor([1]).long()
            graph.val_mask = torch.Tensor([0]).long()
            graph.test_mask = torch.Tensor([0]).long()
            data_train_lst.append(graph)
            assigned = True

        if i in split_idx["valid"]:
            graph.train_mask = torch.Tensor([0]).long()
            graph.val_mask = torch.Tensor([1]).long()
            graph.test_mask = torch.Tensor([0]).long()
            data_val_lst.append(graph)
            assigned = True

        if i in split_idx["test"]:
            graph.train_mask = torch.Tensor([0]).long()
            graph.val_mask = torch.Tensor([0]).long()
            graph.test_mask = torch.Tensor([1]).long()
            data_test_lst.append(graph)
            assigned = True
        if not assigned:
            raise ValueError("Graph not in any split")

    datasets = [
        CustomDataset(data_train_lst),
        CustomDataset(data_val_lst),
        CustomDataset(data_test_lst),
    ]

    return datasets


# Load splits for different dataset
def load_graph_tudataset_split(dataset, cfg):
    r"""Loads the graph dataset with the specified split.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Graph dataset.
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    list
        List containing the train, validation, and test splits.
    """
    # Extract labels from dataset object
    assert (
        len(dataset) > 1
    ), "Torch Geometric TU datasets should have more than one graph in the dataset"
    labels = np.array([data.y.squeeze(0).numpy() for data in dataset])

    if cfg.split_type == "random":
        split_idx = random_splitting(labels, cfg)

    elif cfg.split_type == "k-fold":
        split_idx = k_fold_split(labels, cfg)

    else:
        raise NotImplementedError(
            f"split_type {cfg.split_type} not valid. Choose either 'test' or 'k-fold'"
        )

    train_dataset, val_dataset, test_dataset = (
        assing_train_val_test_mask_to_graphs(dataset, split_idx)
    )

    return [train_dataset, val_dataset, test_dataset]


def load_graph_cocitation_split(dataset, cfg):
    r"""Loads cocitation graph datasets with the specified split.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Graph dataset.
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    list
        List containing the train, validation, and test splits.
    """

    # Extract labels from dataset object
    assert (
        len(dataset) == 1
    ), "Torch Geometric Cocitation dataset should have only one graph"

    data = dataset.data
    labels = data.y.numpy()

    # Ensure labels are one dimensional array
    assert len(labels.shape) == 1, "Labels should be one dimensional array"

    if cfg.split_type == "random":
        splits = random_splitting(labels, cfg)

    elif cfg.split_type == "k-fold":
        splits = k_fold_split(labels, cfg)

    else:
        raise NotImplementedError(
            f"split_type {cfg.split_type} not valid. Choose either 'test' or 'k-fold'"
        )

    # Assign train val test masks to the graph
    data.train_mask = torch.from_numpy(splits["train"])
    data.val_mask = torch.from_numpy(splits["valid"])
    data.test_mask = torch.from_numpy(splits["test"])

    return CustomDataset([data])
