import os.path as osp
import pickle

import numpy as np
import torch
from torch_geometric.data import Data
from torch_sparse import coalesce


def load_citation_dataset(cfg):
    """
    this will read the citation dataset from HyperGCN, and convert it edge_list to
    [[ -V- | -E- ]
     [ -E- | -V- ]]
    """
    print(f"Loading hypergraph dataset from hyperGCN: {dataset}")

    # first load node features:
    with open(osp.join(path, dataset, "features.pickle"), "rb") as f:
        features = pickle.load(f)
        features = features.todense()

    # then load node labels:
    with open(osp.join(path, dataset, "labels.pickle"), "rb") as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f"number of nodes:{num_nodes}, feature dimension: {feature_dim}")

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    with open(osp.join(path, dataset, "hypergraph.pickle"), "rb") as f:
        # hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pickle.load(f)

    print(f"number of hyperedges: {len(hypergraph)}")

    edge_idx = num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    edge_index = np.array([node_list + edge_list, edge_list + node_list], dtype=np.int)
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
    #     n_x = n_expanded
    num_class = len(np.unique(labels.numpy()))
    val_lb = int(n_x * train_percent)
    percls_trn = int(round(train_percent * n_x / num_class))
    # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
    data.n_x = n_x
    # add parameters to attribute

    data.train_percent = train_percent
    data.num_hyperedges = len(hypergraph)

    return data
