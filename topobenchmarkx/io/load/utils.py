import hashlib
import os
import os.path as osp
import pickle

import numpy as np
import omegaconf
import toponetx.datasets.graph as graph
import torch
import torch_geometric
from topomodelx.utils.sparse import from_sparse
from torch_geometric.data import Data
from torch_sparse import coalesce

#from sklearn.model_selection import StratifiedKFold
#from topobenchmarkx.data.datasets import CustomDataset


def get_complex_connectivity(complex, max_rank, signed=False):
    r"""Gets the connectivity matrices for the complex.

    Parameters
    ----------
    complex : topnetx.CellComplex, topnetx.SimplicialComplex
        Cell complex.
    max_rank : int
        Maximum rank of the complex.
    signed : bool
        If True, returns signed connectivity matrices.

    Returns
    -------
    dict
        Dictionary containing the connectivity matrices.
    """
    practical_shape = list(
        np.pad(list(complex.shape), (0, max_rank + 1 - len(complex.shape)))
    )
    connectivity = {}
    for rank_idx in range(max_rank + 1):
        for connectivity_info in [
            "incidence",
            "down_laplacian",
            "up_laplacian",
            "adjacency",
            "hodge_laplacian",
        ]:
            try:
                connectivity[f"{connectivity_info}_{rank_idx}"] = from_sparse(
                    getattr(complex, f"{connectivity_info}_matrix")(
                        rank=rank_idx, signed=signed
                    )
                )
            except ValueError:
                if connectivity_info == "incidence":
                    connectivity[
                        f"{connectivity_info}_{rank_idx}"
                    ] = generate_zero_sparse_connectivity(
                        m=practical_shape[rank_idx - 1], n=practical_shape[rank_idx]
                    )
                else:
                    connectivity[
                        f"{connectivity_info}_{rank_idx}"
                    ] = generate_zero_sparse_connectivity(
                        m=practical_shape[rank_idx], n=practical_shape[rank_idx]
                    )
    connectivity["shape"] = practical_shape
    return connectivity


def generate_zero_sparse_connectivity(m, n):
    r"""Generates a zero sparse connectivity matrix.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.

    Returns
    -------
    torch.sparse_coo_tensor
        Zero sparse connectivity matrix.
    """
    return torch.sparse_coo_tensor((m, n)).coalesce()


def load_cell_complex_dataset(cfg):
    r"""Loads cell complex datasets."""
    pass


def load_simplicial_dataset(cfg):
    r"""Loads simplicial datasets.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    torch_geometric.data.Data
        Simplicial dataset.
    """
    if cfg["data_name"] != "KarateClub":
        return NotImplementedError
    data = graph.karate_club(complex_type="simplicial", feat_dim=2)
    max_rank = data.dim
    features = {}
    dict_feat_equivalence = {
        0: "node_feat",
        1: "edge_feat",
        2: "face_feat",
        3: "tetrahedron_feat",
    }
    for rank_idx in range(max_rank + 1):
        try:
            features["x_{}".format(rank_idx)] = torch.tensor(
                np.stack(
                    list(
                        data.get_simplex_attributes(
                            dict_feat_equivalence[rank_idx]
                        ).values()
                    )
                )
            )
        except:
            features["x_{}".format(rank_idx)] = torch.tensor(
                np.zeros((data.shape[rank_idx], 0))
            )
    features["y"] = torch.tensor(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    # features['num_nodes'] = data.shape[0]
    features["x"] = features["x_0"]
    connectivity = get_complex_connectivity(data, max_rank)
    data = torch_geometric.data.Data(**connectivity, **features)

    # Project node-level features to edge-level (WHY DO WE NEED IT, data already has x_1)
    data.x_1 = data.x_1 + torch.mm(data.incidence_1.to_dense().T, data.x_0)

    # TODO: Fix the splits
    data = torch_geometric.transforms.random_node_split.RandomNodeSplit(
        num_val=4, num_test=4
    )(data)
    return data


def load_hypergraph_pickle_dataset(cfg):
    r"""Loads hypergraph datasets from pickle files.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    torch_geometric.data.Data
        Hypergraph dataset.
    """
    data_dir = cfg["data_dir"]
    print(f"Loading {cfg['data_domain']} dataset name: {cfg['data_name']}")

    # Load node features:

    with open(osp.join(data_dir, "features.pickle"), "rb") as f:
        features = pickle.load(f)
        features = features.todense()

    # Load node labels:
    with open(osp.join(data_dir, "labels.pickle"), "rb") as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f"number of nodes:{num_nodes}, feature dimension: {feature_dim}")

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # Load hypergraph.
    with open(osp.join(data_dir, "hypergraph.pickle"), "rb") as f:
        # Hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pickle.load(f)

    print(f"number of hyperedges: {len(hypergraph)}")

    edge_idx = 0  # num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    # check that every node is in some hyperedge
    if len(np.unique(node_list)) != num_nodes:
        # add self hyperedges to isolated nodes
        isolated_nodes = np.setdiff1d(np.arange(num_nodes), np.unique(node_list))

        for node in isolated_nodes:
            node_list += [node]
            edge_list += [edge_idx]
            edge_idx += 1
            hypergraph[f"Unique_additonal_he_{edge_idx}"] = [node]

    edge_index = np.array([node_list, edge_list], dtype=int)
    edge_index = torch.LongTensor(edge_index)

    data = Data(
        x=features,
        x_0=features,
        edge_index=edge_index,
        incidence_hyperedges=edge_index,
        y=labels,
    )

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(
        data.edge_index, None, total_num_node_id_he_id, total_num_node_id_he_id
    )

    n_x = num_nodes
    num_class = len(np.unique(labels.numpy()))

    # Add parameters to attribute
    data.n_x = n_x
    data.num_hyperedges = len(hypergraph)
    data.num_class = num_class

    data.incidence_hyperedges = torch.sparse_coo_tensor(
        data.edge_index,
        values=torch.ones(data.edge_index.shape[1]),
        size=(data.num_nodes, data.num_hyperedges),
    )

    # Print some info
    print("Final num_hyperedges", data.num_hyperedges)
    print("Final num_nodes", data.num_nodes)
    print("Final num_class", data.num_class)

    return data


def get_Planetoid_pyg(cfg):
    r"""Loads Planetoid graph datasets from torch_geometric.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    torch_geometric.data.Data
        Graph dataset.
    """
    data_dir, data_name = cfg["data_dir"], cfg["data_name"]
    dataset = torch_geometric.datasets.Planetoid(data_dir, data_name)
    data = dataset.data
    data.num_nodes = data.x.shape[0]
    return data


def get_TUDataset_pyg(cfg):
    r"""Loads TU graph datasets from torch_geometric.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    list
        List containing the graph dataset.
    """
    data_dir, data_name = cfg["data_dir"], cfg["data_name"]
    dataset = torch_geometric.datasets.TUDataset(root=data_dir, name=data_name)
    data_lst = [data for data in dataset]
    return data_lst


def ensure_serializable(obj):
    r"""Ensures that the object is serializable.

    Parameters
    ----------
    obj : object
        Object to ensure serializability.

    Returns
    -------
    object
        Object that is serializable.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = ensure_serializable(value)
        return obj
    elif isinstance(obj, (list, tuple)):
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return {ensure_serializable(item) for item in obj}
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, omegaconf.dictconfig.DictConfig):
        return dict(obj)
    else:
        return None


def make_hash(o):
    r"""Makes a hash from a dictionary, list, tuple or set to any level, that
    contains only other hashable types (including any lists, tuples, sets, and
    dictionaries).

    Parameters
    ----------
    o : dict, list, tuple, set
        Object to hash.

    Returns
    -------
    int
        Hash of the object.
    """
    sha1 = hashlib.sha1()
    sha1.update(str.encode(str(o)))
    hash_as_hex = sha1.hexdigest()
    # convert the hex back to int and restrict it to the relevant int range
    seed = int(hash_as_hex, 16) % 4294967295
    return seed

# Moved to splits_utils.py
# def load_split(data, cfg):
#     r"""Loads the split for the graph dataset.

#     Parameters
#     ----------
#     data : torch_geometric.data.Data
#         Graph dataset.
#     cfg : DictConfig
#         Configuration parameters.

#     Returns
#     -------
#     torch_geometric.data.Data
#         Graph dataset with the specified split.
#     """
#     data_dir = os.path.join(cfg["data_split_dir"], "train_prop=0.5")
#     load_path = f"{data_dir}/split_{cfg['data_seed']}.npz"
#     splits = np.load(load_path, allow_pickle=True)
#     # Upload masks
#     data.train_mask = torch.from_numpy(splits["train"])
#     data.val_mask = torch.from_numpy(splits["valid"])
#     data.test_mask = torch.from_numpy(splits["test"])

#     # Check that all nodes assigned to splits
#     assert (
#         torch.unique(
#             torch.concat([data.train_mask, data.val_mask, data.test_mask])
#         ).shape[0]
#         == data.num_nodes
#     ), "Not all nodes within splits"
#     return data


# def k_fold_split(dataset, parameters, ignore_negative=True):
#     """
#     Returns train and valid indices as in K-Fold Cross-Validation. If the split already exists
#     it loads it automatically, otherwise it creates the split file for the subsequent runs.

#     Parameters
#     ----------
#     dataset : torch_geometric.data.Dataset
#         Graph dataset.
#     parameters : DictConfig
#         Configuration parameters.
#     ignore_negative : bool
#         If True, ignores negative labels.

#     Returns
#     -------
#     dict
#         Dictionary containing the train, validation and test indices.
#     """
#     data_dir = parameters.data_split_dir
#     k = parameters.k
#     fold = parameters.data_seed
#     assert fold < k, "data_seed needs to be less than k"

#     torch.manual_seed(0)
#     np.random.seed(0)

#     split_dir = os.path.join(data_dir, f"{k}-fold")
#     if not os.path.isdir(split_dir):
#         os.makedirs(split_dir)
#     split_path = os.path.join(split_dir, f"{fold}.npz")
#     if os.path.isfile(split_path):
#         split_idx = np.load(split_path)
#         return split_idx
#     else:
#         if parameters.task_level == "graph":
#             labels = dataset.y
#         else:
#             # TODO: change if to assert in case comment below is positively resolved
#             if len(dataset) == 1:
#                 labels = dataset.y
#             # TODO: Do we actually need this? I don't recall any node level dataset with multiple graphs
#             # else:
                
#             #     # This is the case of node level task with multiple graphs
#             #     # Here dataset.y cannot be used to measure the number of elements to associate to the splits
#             #     labels = torch.ones(len(dataset))

#         # TODO: I dont remember any dataset which actually has -1 as label? should we remove this if statement?
#         if ignore_negative:
#             labeled_nodes = torch.where(labels != -1)[0]
#         else:
#             labeled_nodes = labels

#         n = labeled_nodes.shape[0]

#         if len(dataset) == 1:
#             y = dataset[0].y.squeeze(0).numpy()
#         else:
#             y = np.array([data.y.squeeze(0).numpy() for data in dataset])

#         x_idx = np.arange(n)
#         x_idx = np.random.permutation(x_idx)
#         y = y[x_idx]

#         skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

#         for fold_n, (train_idx, valid_idx) in enumerate(skf.split(x_idx, y)):
#             split_idx = {"train": train_idx, "valid": valid_idx, "test": valid_idx}

#             assert np.all(
#                 np.sort(
#                     np.array(split_idx["train"].tolist() + split_idx["valid"].tolist())
#                 )
#                 == np.sort(np.arange(len(labels)))
#             ), "Not every sample has been loaded."
#             split_path = os.path.join(split_dir, f"{fold_n}.npz")

#             np.savez(split_path, **split_idx)

#     split_path = os.path.join(split_dir, f"{fold}.npz")
#     split_idx = np.load(split_path)
#     return split_idx


# def load_graph_cocitation_split(dataset, cfg):
#     r"""Loads cocitation graph datasets with the specified split.

#     Parameters
#     ----------
#     dataset : torch_geometric.data.Dataset
#         Graph dataset.
#     cfg : DictConfig
#         Configuration parameters.

#     Returns
#     -------
#     list
#         List containing the train, validation, and test splits.
#     """
#     data = dataset.data
    
#     if cfg.split_type == "test":
#         data = load_split(data, cfg)
#         return CustomDataset([data])
    
#     elif cfg.split_type == "k-fold":
#         split_idx = k_fold_split(dataset, cfg)
#         data.train_mask = split_idx["train"]
#         data.val_mask = split_idx["valid"]
#         data.test_mask = split_idx["test"]
#         return CustomDataset([data])
    
#     else:
#         raise NotImplementedError(
#             f"split_type {cfg.split_type} not valid. Choose either 'test' or 'k-fold'"
#         )


# def rand_train_test_idx(
#     label, train_prop=0.5, valid_prop=0.25, ignore_negative=True, balance=False, seed=0
# ):
#     """Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks
#     randomly splits label into train/valid/test splits.

#     Parameters
#     ----------
#     label : torch.Tensor
#         Label tensor.
#     train_prop : float
#         Proportion of training data.
#     valid_prop : float
#         Proportion of validation data.
#     ignore_negative : bool
#         If True, ignores negative labels.
#     balance : bool
#         If True, balances the classes.
#     seed : int
#         Random seed.

#     Returns
#     -------
#     dict
#         Dictionary containing the train, validation and test indices.
#     """
#     # set seed
#     torch.manual_seed(seed)
#     np.random.seed(seed)

#     if not balance:
#         if ignore_negative:
#             labeled_nodes = torch.where(label != -1)[0]
#         else:
#             labeled_nodes = label

#         n = labeled_nodes.shape[0]
#         train_num = int(n * train_prop)
#         valid_num = int(n * valid_prop)

#         perm = torch.as_tensor(np.random.permutation(n))

#         train_indices = perm[:train_num]
#         val_indices = perm[train_num : train_num + valid_num]
#         test_indices = perm[train_num + valid_num :]

#         if not ignore_negative:
#             return train_indices, val_indices, test_indices

#         train_idx = labeled_nodes[train_indices]
#         valid_idx = labeled_nodes[val_indices]
#         test_idx = labeled_nodes[test_indices]

#         split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
#     else:
#         #         ipdb.set_trace()
#         indices = []
#         for i in range(label.max() + 1):
#             index = torch.where((label == i))[0].view(-1)
#             index = index[torch.randperm(index.size(0))]
#             indices.append(index)

#         percls_trn = int(train_prop / (label.max() + 1) * len(label))
#         val_lb = int(valid_prop * len(label))
#         train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
#         rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
#         rest_index = rest_index[torch.randperm(rest_index.size(0))]
#         valid_idx = rest_index[:val_lb]
#         test_idx = rest_index[val_lb:]
#         split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}

#     # Save splits to disk

#     return split_idx


# def get_train_val_test_graph_datasets(dataset, split_idx):
#     r"""Splits the graph dataset into train, validation, and test datasets.

#     Parameters
#     ----------
#     dataset : torch_geometric.data.Dataset
#         Graph dataset.
#     split_idx : dict
#         Dictionary containing the indices for the train, validation, and test splits.

#     Returns
#     -------
#     list
#         List containing the train, validation, and test datasets.
#     """
#     data_train_lst, data_val_lst, data_test_lst = [], [], []

#     # Go over each of the graph and assign correct label
#     for i in range(len(dataset)):
#         graph = dataset[i]
#         assigned = False
#         if i in split_idx["train"]:
#             graph.train_mask = torch.Tensor([1]).long()
#             graph.val_mask = torch.Tensor([0]).long()
#             graph.test_mask = torch.Tensor([0]).long()
#             data_train_lst.append(graph)
#             assigned = True

#         if i in split_idx["valid"]:
#             graph.train_mask = torch.Tensor([0]).long()
#             graph.val_mask = torch.Tensor([1]).long()
#             graph.test_mask = torch.Tensor([0]).long()
#             data_val_lst.append(graph)
#             assigned = True

#         if i in split_idx["test"]:
#             graph.train_mask = torch.Tensor([0]).long()
#             graph.val_mask = torch.Tensor([0]).long()
#             graph.test_mask = torch.Tensor([1]).long()
#             data_test_lst.append(graph)
#             assigned = True
#         if not assigned:
#             raise ValueError("Graph not in any split")

#     datasets = [
#         CustomDataset(data_train_lst),
#         CustomDataset(data_val_lst),
#         CustomDataset(data_test_lst),
#     ]

#     return datasets


# def load_graph_tudataset_split(dataset, cfg):
#     r"""Loads the graph dataset with the specified split.

#     Parameters
#     ----------
#     dataset : torch_geometric.data.Dataset
#         Graph dataset.
#     cfg : DictConfig
#         Configuration parameters.

#     Returns
#     -------
#     list
#         List containing the train, validation, and test splits.
#     """
#     if cfg.split_type == "test":
#         labels = dataset.y
#         split_idx = rand_train_test_idx(labels)
#     elif cfg.split_type == "k-fold":
#         split_idx = k_fold_split(dataset, cfg)
#     else:
#         raise NotImplementedError(
#             f"split_type {cfg.split_type} not valid. Choose either 'test' or 'k-fold'"
#         )

#     train_dataset, val_dataset, test_dataset = get_train_val_test_graph_datasets(
#         dataset, split_idx
#     )

#     return [train_dataset, val_dataset, test_dataset]