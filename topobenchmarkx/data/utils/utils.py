import hashlib
import networkx as nx
import numpy as np
import omegaconf
import toponetx.datasets.graph as graph
import torch
import torch_geometric
from topomodelx.utils.sparse import from_sparse

def get_complex_connectivity(complex, max_rank, signed=False):
    """
    Gets the connectivity matrices for the complex.

    Parameters
    ----------
    complex : topnetx.CellComplex or topnetx.SimplicialComplex
        Cell or simplicial complex.
    max_rank : int
        Maximum rank of the complex.
    signed : bool, optional
        If True, returns signed connectivity matrices. Default is False.

    Returns
    -------
    dict
        Dictionary containing the connectivity matrices.
    """
    practical_shape = list(np.pad(list(complex.shape), (0, max_rank + 1 - len(complex.shape))))
    connectivity = {}
    for rank_idx in range(max_rank + 1):
        for connectivity_info in ["incidence", "down_laplacian", "up_laplacian", "adjacency", "hodge_laplacian"]:
            try:
                connectivity[f"{connectivity_info}_{rank_idx}"] = from_sparse(
                    getattr(complex, f"{connectivity_info}_matrix")(rank=rank_idx, signed=signed)
                )
            except ValueError:
                if connectivity_info == "incidence":
                    connectivity[f"{connectivity_info}_{rank_idx}"] = generate_zero_sparse_connectivity(
                        m=practical_shape[rank_idx - 1], n=practical_shape[rank_idx]
                    )
                else:
                    connectivity[f"{connectivity_info}_{rank_idx}"] = generate_zero_sparse_connectivity(
                        m=practical_shape[rank_idx], n=practical_shape[rank_idx]
                    )
    connectivity["shape"] = practical_shape
    return connectivity

def generate_zero_sparse_connectivity(m, n):
    """
    Generates a zero sparse connectivity matrix.

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
    """
    Loads cell complex datasets.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    torch_geometric.data.Data
        Cell complex dataset.

    Raises
    ------
    NotImplementedError
        Always raises this error as this function is not yet implemented.
    """
    raise NotImplementedError

def load_simplicial_dataset(cfg):
    """
    Loads simplicial datasets.

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
    dict_feat_equivalence = {0: "node_feat", 1: "edge_feat", 2: "face_feat", 3: "tetrahedron_feat"}
    for rank_idx in range(max_rank + 1):
        try:
            features[f"x_{rank_idx}"] = torch.tensor(
                np.stack(list(data.get_simplex_attributes(dict_feat_equivalence[rank_idx]).values()))
            )
        except ValueError:
            features[f"x_{rank_idx}"] = torch.tensor(np.zeros((data.shape[rank_idx], 0)))
    features["y"] = torch.tensor(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    features["x"] = features["x_0"]
    connectivity = get_complex_connectivity(data, max_rank)
    data = torch_geometric.data.Data(**connectivity, **features)
    data.x_1 = data.x_1 + torch.mm(data.incidence_1.to_dense().T, data.x_0)
    return torch_geometric.transforms.random_node_split.RandomNodeSplit(num_val=4, num_test=4)(data)

def load_manual_graph():
    """
    Create a manual graph for testing purposes.

    Returns
    -------
    torch_geometric.data.Data
        Manually created graph dataset.
    """
    vertices = [i for i in range(8)]
    y = [0, 1, 1, 1, 0, 0, 0, 0]
    edges = [
        [0, 1], [0, 2], [0, 4], [1, 2], [2, 3], [5, 2], [5, 6], [6, 3], [5, 7], [2, 7], [0, 7]
    ]
    tetrahedrons = [[0, 1, 2, 4]]
    for tetrahedron in tetrahedrons:
        for i in range(len(tetrahedron)):
            for j in range(i + 1, len(tetrahedron)):
                edges.append([tetrahedron[i], tetrahedron[j]])
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    G.to_undirected()
    edge_list = torch.Tensor(list(G.edges())).T.long()
    x = torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000]).unsqueeze(1).float()
    return torch_geometric.data.Data(x=x, edge_index=edge_list, num_nodes=len(vertices), y=torch.tensor(y))

def ensure_serializable(obj):
    """
    Ensures that the object is serializable.

    Parameters
    ----------
    obj : object
        Object to ensure serializability.

    Returns
    -------
    object
        Serializable object.
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
    """
    Makes a hash from a dictionary, list, tuple or set to any level, that contains only other hashable types.

    Parameters
    ----------
    o : dict, list, tuple, or set
        Object to hash.

    Returns
    -------
    int
        Hash of the object.
    """
    sha1 = hashlib.sha1()
    sha1.update(str.encode(str(o)))
    hash_as_hex = sha1.hexdigest()
    return int(hash_as_hex, 16) % 4294967295
