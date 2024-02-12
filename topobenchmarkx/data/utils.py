import hashlib
import os.path as osp
import pickle

import numpy as np
import toponetx.datasets.graph as graph
import torch
import torch_geometric
from topomodelx.utils.sparse import from_sparse
from torch_geometric.data import Data
from torch_sparse import coalesce


def load_cell_complex_dataset(cfg):
    pass


def load_simplicial_dataset(cfg):
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
    connectivity = {}
    connectivity.update(
        {
            "incidence_{}".format(rank_idx): from_sparse(
                data.incidence_matrix(rank_idx)
            )
            for rank_idx in range(1, max_rank)
        }
    )
    for rank_idx in range(max_rank + 1):
        self_adjacency = 0
        try:
            laplacian_down = from_sparse(data.down_laplacian_matrix(rank=rank_idx))
            self_adjacency += 1
        except ValueError:
            laplacian_down = torch.zeros(
                (data.shape[rank_idx], data.shape[rank_idx])
            ).to_sparse()
        # Up laplacian
        try:
            laplacian_up = from_sparse(data.up_laplacian_matrix(rank=rank_idx))
            self_adjacency += 1
        except ValueError:
            laplacian_up = torch.zeros(
                (data.shape[rank_idx], data.shape[rank_idx])
            ).to_sparse()
        adjacency = (
            laplacian_down
            + laplacian_up
            + self_adjacency * torch.eye(data.shape[rank_idx]).to_sparse()
        )
        connectivity.update({"laplacian_down_{}".format(rank_idx): laplacian_down})
        connectivity.update({"laplacian_up_{}".format(rank_idx): laplacian_up})
        connectivity.update({"adjacency_{}".format(rank_idx): adjacency})
    connectivity.update({"shape": data.shape})
    data = torch_geometric.data.Data(**connectivity, **features)

    # Project node-level features to edge-level (WHY DO WE NEED IT, data already has x_1)
    data.x_1 = data.x_1 + torch.mm(data.incidence_1.to_dense().T, data.x_0)

    # TODO: Fix the splits
    data = torch_geometric.transforms.random_node_split.RandomNodeSplit(
        num_val=4, num_test=4
    )(data)
    return data


def load_hypergraph_pickle_dataset(cfg):
    """
    this will read the citation dataset from HyperGCN, and convert it edge_list to
    [[ -V- ]]
     [ -E- ]]
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

    data = Data(x=features, edge_index=edge_index, y=labels)

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

    data.edge_index = torch.sparse_coo_tensor(
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
    data_dir, data_name = cfg["data_dir"], cfg["data_name"]
    dataset = torch_geometric.datasets.Planetoid(data_dir, data_name)
    data = dataset.data
    data.num_nodes = data.x.shape[0]
    return data


def get_TUDataset_pyg(cfg):
    data_dir, data_name = cfg["data_dir"], cfg["data_name"]
    dataset = torch_geometric.datasets.TUDataset(root=data_dir, name=data_name)
    data_lst = [data for data in dataset]
    return data_lst


def load_split(data, cfg):
    data_dir = cfg["data_split_dir"]
    load_path = f"{data_dir}/split_{cfg['data_seed']}.npz"
    splits = np.load(load_path, allow_pickle=True)
    data.train_mask = torch.from_numpy(splits["train"])
    data.val_mask = torch.from_numpy(splits["valid"])
    data.test_mask = torch.from_numpy(splits["test"])

    # check that all nodes belong to splits
    assert (
        torch.unique(
            torch.concat([data.train_mask, data.val_mask, data.test_mask])
        ).shape[0]
        == data.num_nodes
    ), "Not all nodes within splits"
    return data


def make_hash(o):
    """
    Makes a hash from a dictionary, list, tuple or set to any level, that contains
    only other hashable types (including any lists, tuples, sets, and
    dictionaries).
    """
    sha1 = hashlib.sha1()
    sha1.update(str.encode(str(o)))
    hash_as_hex = sha1.hexdigest()
    # convert the hex back to int and restrict it to the relevant int range
    seed = int(hash_as_hex, 16) % 4294967295
    return seed
    # if isinstance(o, (set, tuple, list)):
    #     return tuple([make_hash(e) for e in o])

    # elif not isinstance(o, dict):
    #     return hash(o)

    # new_o = copy.deepcopy(o)
    # for k, v in new_o.items():
    #     new_o[k] = make_hash(v)

    # return hash(tuple(frozenset(sorted(new_o.items()))))
