"""Split utilities."""

import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from topobenchmark.dataloader import DataloadDataset


# Generate splits in different fasions
def k_fold_split(labels, parameters):
    """Return train and valid indices as in K-Fold Cross-Validation.

    If the split already exists it loads it automatically, otherwise it creates the
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
        Dictionary containing the train, validation and test indices, with keys "train", "valid", and "test".
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
    r"""Randomly splits label into train/valid/test splits.

    Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor.
    parameters : DictConfig
        Configuration parameter.
    global_data_seed : int
        Seed for the random number generator.

    Returns
    -------
    dict:
        Dictionary containing the train, validation and test indices with keys "train", "valid", and "test".
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


def assign_train_val_test_mask_to_graphs(dataset, split_idx):
    r"""Split the graph dataset into train, validation, and test datasets.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Considered dataset.
    split_idx : dict
        Dictionary containing the train, validation, and test indices.

    Returns
    -------
    list:
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

    return (
        DataloadDataset(data_train_lst),
        DataloadDataset(data_val_lst),
        DataloadDataset(data_test_lst),
    )


def load_transductive_splits(dataset, parameters):
    r"""Load the graph dataset with the specified split.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Graph dataset.
    parameters : DictConfig
        Configuration parameters.

    Returns
    -------
    list:
        List containing the train, validation, and test splits.
    """
    # Extract labels from dataset object
    assert (
        len(dataset) == 1
    ), "Dataset should have only one graph in a transductive setting."

    data = dataset.data_list[0]
    labels = data.y.numpy()

    # Ensure labels are one dimensional array
    assert len(labels.shape) == 1, "Labels should be one dimensional array"

    if parameters.split_type == "random":
        splits = random_splitting(labels, parameters)

    elif parameters.split_type == "k-fold":
        splits = k_fold_split(labels, parameters)

    else:
        raise NotImplementedError(
            f"split_type {parameters.split_type} not valid. Choose either 'random' or 'k-fold'"
        )

    # Assign train val test masks to the graph
    data.train_mask = torch.from_numpy(splits["train"])
    data.val_mask = torch.from_numpy(splits["valid"])
    data.test_mask = torch.from_numpy(splits["test"])

    if parameters.get("standardize", False):
        # Standardize the node features respecting train mask
        data.x = (data.x - data.x[data.train_mask].mean(0)) / data.x[
            data.train_mask
        ].std(0)
        data.y = (data.y - data.y[data.train_mask].mean(0)) / data.y[
            data.train_mask
        ].std(0)

    return DataloadDataset([data]), None, None


def load_inductive_splits(dataset, parameters):
    r"""Load multiple-graph datasets with the specified split.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Graph dataset.
    parameters : DictConfig
        Configuration parameters.

    Returns
    -------
    list:
        List containing the train, validation, and test splits.
    """
    # Extract labels from dataset object
    assert (
        len(dataset) > 1
    ), "Datasets should have more than one graph in an inductive setting."

    # Handle OnDiskDataset case
    if hasattr(dataset, "dataset"):
        # Get total number of rows from SQLite database
        total_rows = len(dataset)
        # I don't think the labels matter, but rather how many pairs there are...
        # TODO: Don't think this assumption is legal
        labels = np.arange(total_rows)
    else:
        labels = np.array(
            [data.y.squeeze(0).numpy() for data in dataset.data_list]
        )

    if parameters.split_type == "random":
        split_idx = random_splitting(labels, parameters)

    elif parameters.split_type == "k-fold":
        split_idx = k_fold_split(labels, parameters)

    elif parameters.split_type == "fixed" and hasattr(dataset, "split_idx"):
        split_idx = dataset.split_idx

    else:
        raise NotImplementedError(
            f"split_type {parameters.split_type} not valid. Choose either 'random', 'k-fold' or 'fixed'.\
            If 'fixed' is chosen, the dataset should have the attribute split_idx"
        )

    train_dataset, val_dataset, test_dataset = (
        assign_train_val_test_mask_to_graphs(dataset, split_idx)
    )

    return train_dataset, val_dataset, test_dataset


def load_inductive_split_indices(dataset, parameters):
    r"""Load multiple-graph dataset indices with the specified split.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Graph dataset.
    parameters : DictConfig
        Configuration parameters.

    Returns
    -------
    list:
        List containing the train, validation, and test split indices.
    """
    # Extract labels from dataset object
    assert (
        len(dataset) > 1
    ), "Datasets should have more than one graph in an inductive setting."

    # Handle OnDiskDataset case
    if hasattr(dataset, "dataset"):
        # Get total number of rows from SQLite database
        total_rows = len(dataset)
        # I don't think the labels matter, but rather how many pairs there are...
        # TODO: Don't think this assumption is legal
        labels = np.arange(total_rows)
    else:
        labels = np.array(
            [data.y.squeeze(0).numpy() for data in dataset.data_list]
        )

    if parameters.split_type == "random":
        split_idx = random_splitting(labels, parameters)

    elif parameters.split_type == "fixed" and hasattr(dataset, "split_idx"):
        split_idx = dataset.split_idx

    else:
        raise NotImplementedError(
            f"split_type {parameters.split_type} not valid. Choose either 'random' or 'fixed'. \
            'k-fold' is not yet implemented. \
            If 'fixed' is chosen, the dataset should have the attribute split_idx"
        )

    return (
        split_idx["train"],
        split_idx.get("valid", None),
        split_idx.get("test", None),
    )


def load_coauthorship_hypergraph_splits(data, parameters, train_prop=0.5):
    r"""Load the split generated by rand_train_test_idx function.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph dataset.
    parameters : DictConfig
        Configuration parameters.
    train_prop : float
        Proportion of training data.

    Returns
    -------
    torch_geometric.data.Data:
        Graph dataset with the specified split.
    """

    data_dir = os.path.join(
        parameters["data_split_dir"], f"train_prop={train_prop}"
    )
    load_path = f"{data_dir}/split_{parameters['data_seed']}.npz"
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
    return DataloadDataset([data]), None, None
