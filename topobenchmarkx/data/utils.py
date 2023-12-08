import os.path as osp
import pickle

import numpy as np
import torch
from torch_geometric.data import Data
from torch_sparse import coalesce


def load_hypergraph_coauthorhip_dataset(cfg):
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

    edge_idx = num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

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

    data.edge_index = torch.sparse_coo_tensor(
        data.edge_index,
        values=torch.ones(data.edge_index.shape[1]),
        size=(data.num_nodes, data.num_edges),
    )

    n_x = num_nodes
    num_class = len(np.unique(labels.numpy()))

    # Add parameters to attribute
    data.n_x = n_x
    data.num_hyperedges = len(hypergraph)
    data.num_class = num_class
    return data
