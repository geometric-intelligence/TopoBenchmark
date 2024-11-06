"""Data utilities."""

import hashlib

import networkx as nx
import numpy as np
import omegaconf
import toponetx.datasets.graph as graph
import torch
import torch_geometric
from topomodelx.utils.sparse import from_sparse


def get_routes_from_neighborhoods(neighborhoods):
    """Get the routes from the neighborhoods.

    Combination of src_rank, dst_rank. ex: [[0, 0], [1, 0], [1, 1], [1, 1], [2, 1]].

    Parameters
    ----------
    neighborhoods : list
        List of neighborhoods of interest.

    Returns
    -------
    list
        List of routes.
    """
    routes = []
    for neighborhood in neighborhoods:
        split = neighborhood.split("-")
        src_rank = int(split[-1])
        r = int(split[0]) if len(split) == 3 else 1
        route = (
            [src_rank, src_rank - r]
            if "down" in neighborhood
            else [src_rank, src_rank + r]
        )
        routes.append(route)
    return routes


def get_complex_connectivity(
    complex, max_rank, neighborhoods=None, signed=False
):
    """Get the connectivity matrices for the complex.

    Parameters
    ----------
    complex : toponetx.CellComplex or toponetx.SimplicialComplex
        Cell complex.
    max_rank : int
        Maximum rank of the complex.
    neighborhoods : list, optional
        List of neighborhoods of interest.
    signed : bool, optional
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
            "coadjacency",
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
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx - 1],
                            n=practical_shape[rank_idx],
                        )
                    )
                else:
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx],
                            n=practical_shape[rank_idx],
                        )
                    )
    if neighborhoods is not None:
        connectivity = select_neighborhoods_of_interest(
            connectivity, neighborhoods
        )
    connectivity["shape"] = practical_shape
    return connectivity


def select_neighborhoods_of_interest(connectivity, neighborhoods):
    """Select the neighborhoods of interest.

    Parameters
    ----------
    connectivity : dict
        Connectivity matrices generated by default.
    neighborhoods : list
        List of neighborhoods of interest.

    Returns
    -------
    dict
        Connectivity matrices of interest.
    """

    def generate_adjacency_from_laplacian(sparse_tensor):
        """Generate an adjacency matrix from a Laplacian matrix.

        Parameters
        ----------
        sparse_tensor : torch.sparse_coo_tensor
            Sparse tensor representing the Laplacian matrix.

        Returns
        -------
        torch.sparse_coo_tensor
            Sparse tensor representing the adjacency matrix.
        """
        indices = sparse_tensor._indices()
        values = sparse_tensor._values()

        # Create a mask for non-diagonal elements
        mask = indices[0] != indices[1]

        # Filter indices and values based on the mask
        new_indices = indices[:, mask]
        new_values = values[mask]

        # Turn values to 1s
        new_values = new_values / new_values

        # Construct a new sparse tensor
        return torch.sparse_coo_tensor(
            new_indices, new_values, sparse_tensor.size()
        )

    useful_connectivity = {}
    for neighborhood in neighborhoods:
        src_rank = int(neighborhood.split("-")[-1])
        try:
            if len(neighborhood.split("-")) == 2:
                r = 1
                neighborhood_type = neighborhood.split("-")[0]
                if "adjacency" in neighborhood_type:
                    useful_connectivity[neighborhood] = (
                        connectivity[f"adjacency_{src_rank}"]
                        if "up" in neighborhood_type
                        else connectivity[f"coadjacency_{src_rank}"]
                    )
                elif "laplacian" in neighborhood_type:
                    useful_connectivity[neighborhood] = connectivity[
                        f"{neighborhood_type}_{src_rank}"
                    ]
                elif "incidence" in neighborhood_type:
                    useful_connectivity[neighborhood] = (
                        connectivity[f"incidence_{src_rank+1}"].T
                        if "up" in neighborhood_type
                        else connectivity[f"incidence_{src_rank}"]
                    )
            elif len(neighborhood.split("-")) == 3:
                r = int(neighborhood.split("-")[0])
                neighborhood_type = neighborhood.split("-")[1]
                if (
                    "adjacency" in neighborhood_type
                    or "laplacian" in neighborhood_type
                ):
                    direction, connectivity_type = neighborhood_type.split("_")
                    if direction == "up":
                        # Multiply consecutive incidence matrices up to getting the desired rank
                        matrix = torch.sparse.mm(
                            connectivity[f"incidence_{src_rank+1}"],
                            connectivity[f"incidence_{src_rank+2}"],
                        )
                        for idx in range(src_rank + 3, src_rank + r + 1):
                            matrix = torch.sparse.mm(
                                matrix, connectivity[f"incidence_{idx}"]
                            )
                        # Multiply the resulting matrix by its transpose to get the laplacian matrix
                        matrix = torch.sparse.mm(matrix, matrix.T)
                        # Turn all values to 1s
                        matrix = torch.sparse_coo_tensor(
                            matrix.indices(),
                            matrix.values() / matrix.values(),
                            matrix.size(),
                        )
                        # Generate the adjacency matrix from the laplacian if needed
                        useful_connectivity[neighborhood] = (
                            generate_adjacency_from_laplacian(matrix)
                            if "adjacency" in neighborhood_type
                            else matrix
                        )
                    elif direction == "down":
                        # Multiply consecutive incidence matrices up to getting the desired rank
                        matrix = torch.sparse.mm(
                            connectivity[f"incidence_{src_rank-r+1}"],
                            connectivity[f"incidence_{src_rank-r+2}"],
                        )
                        for idx in range(src_rank - r + 3, src_rank + 1):
                            matrix = torch.sparse.mm(
                                matrix, connectivity[f"incidence_{idx}"]
                            )
                        # Multiply the resulting matrix by its transpose to get the laplacian matrix
                        matrix = torch.sparse.mm(matrix.T, matrix)
                        # Turn all values to 1s
                        matrix = torch.sparse_coo_tensor(
                            matrix.indices(),
                            matrix.values() / matrix.values(),
                            matrix.size(),
                        )
                        # Generate the adjacency matrix from the laplacian if needed
                        useful_connectivity[neighborhood] = (
                            generate_adjacency_from_laplacian(matrix)
                            if "adjacency" in neighborhood_type
                            else matrix
                        )
                elif "incidence" in neighborhood_type:
                    direction, connectivity_type = neighborhood_type.split("_")
                    if direction == "up":
                        # Multiply consecutive incidence matrices up to getting the desired rank
                        matrix = torch.sparse.mm(
                            connectivity[f"incidence_{src_rank+1}"],
                            connectivity[f"incidence_{src_rank+2}"],
                        )
                        for idx in range(src_rank + 3, src_rank + r + 1):
                            matrix = torch.sparse.mm(
                                matrix, connectivity[f"incidence_{idx}"]
                            )
                        # Turn all values to 1s and transpose the matrix
                        useful_connectivity[neighborhood] = (
                            torch.sparse_coo_tensor(
                                matrix.indices(),
                                matrix.values() / matrix.values(),
                                matrix.size(),
                            ).T
                        )
                    elif direction == "down":
                        # Multiply consecutive incidence matrices up to getting the desired rank
                        matrix = torch.sparse.mm(
                            connectivity[f"incidence_{src_rank-r+1}"],
                            connectivity[f"incidence_{src_rank-r+2}"],
                        )
                        for idx in range(src_rank - r + 3, src_rank + 1):
                            matrix = torch.sparse.mm(
                                matrix, connectivity[f"incidence_{idx}"]
                            )
                        # Turn all values to 1s
                        useful_connectivity[neighborhood] = (
                            torch.sparse_coo_tensor(
                                matrix.indices(),
                                matrix.values() / matrix.values(),
                                matrix.size(),
                            )
                        )
            else:
                useful_connectivity[neighborhood] = connectivity[neighborhood]
        except:  # noqa: E722
            raise ValueError(f"Invalid neighborhood {neighborhood}")  # noqa: B904
    for key in connectivity:
        if "incidence" in key and "-" not in key:
            useful_connectivity[key] = connectivity[key]
    return useful_connectivity


def generate_zero_sparse_connectivity(m, n):
    """Generate a zero sparse connectivity matrix.

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
    r"""Load cell complex datasets.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.
    """
    raise NotImplementedError


def load_simplicial_dataset(cfg):
    """Load simplicial datasets.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    torch_geometric.data.Data
        Simplicial dataset.
    """
    raise NotImplementedError


def load_manual_graph():
    """Create a manual graph for testing purposes.

    Returns
    -------
    torch_geometric.data.Data
        Manual graph.
    """
    # Define the vertices (just 8 vertices)
    vertices = [i for i in range(8)]
    y = [0, 1, 1, 1, 0, 0, 0, 0]
    # Define the edges
    edges = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 2],
        [2, 3],
        [5, 2],
        [5, 6],
        [6, 3],
        [5, 7],
        [2, 7],
        [0, 7],
    ]

    # Define the tetrahedrons
    tetrahedrons = [[0, 1, 2, 4]]

    # Add tetrahedrons
    for tetrahedron in tetrahedrons:
        for i in range(len(tetrahedron)):
            for j in range(i + 1, len(tetrahedron)):
                edges.append([tetrahedron[i], tetrahedron[j]])  # noqa: PERF401

    # Create a graph
    G = nx.Graph()

    # Add vertices
    G.add_nodes_from(vertices)

    # Add edges
    G.add_edges_from(edges)
    G.to_undirected()
    edge_list = torch.Tensor(list(G.edges())).T.long()

    # Generate feature from 0 to 9
    x = torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000]).unsqueeze(1).float()

    return torch_geometric.data.Data(
        x=x,
        edge_index=edge_list,
        num_nodes=len(vertices),
        y=torch.tensor(y),
    )


def ensure_serializable(obj):
    """Ensure that the object is serializable.

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
    elif isinstance(obj, list | tuple):
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return {ensure_serializable(item) for item in obj}
    elif isinstance(obj, str | int | float | bool | type(None)):
        return obj
    elif isinstance(obj, omegaconf.dictconfig.DictConfig):
        return dict(obj)
    else:
        return None


def make_hash(o):
    """Make a hash from a dictionary, list, tuple or set to any level, that contains only other hashable types.

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
    # Convert the hex back to int and restrict it to the relevant int range
    return int(hash_as_hex, 16) % 4294967295
