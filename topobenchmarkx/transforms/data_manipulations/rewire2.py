import numpy as np
import torch
import torch_geometric
from sklearn.mixture import GaussianMixture


def _find_threshold(curv_vals: np.ndarray) -> float:
    """
    Model the curvature distribution with a mixture of two Gaussians.
    Find the midpoint between the means of the two Gaussians.
    """
    gmm = GaussianMixture(n_components=2, random_state=0).fit(curv_vals)

    mean1 = gmm.means_[0][0]
    std1 = np.sqrt(gmm.covariances_[0][0][0])

    mean2 = gmm.means_[1][0]
    std2 = np.sqrt(gmm.covariances_[1][0][0])

    # Determine which mean is smaller and assign mean_min and mean_max
    if mean1 < mean2:
        mean_min = mean1
        mean_max = mean2

        std_min = std1
        std_max = std2
    else:
        mean_min = mean2
        mean_max = mean1

        std_min = std2
        std_max = std1

    assert mean_min < mean_max, "Something wrong with the means."

    return (mean_min, mean_max, std_min, std_max)


class AFRCN(torch_geometric.transforms.BaseTransform):
    r"""Adds the node degree as one hot encodings to the node features.

    Parameters
    ----------
    max_degree : int
        The maximum degree of the graph.
    cat : bool, optional
        If set to `True`, the one hot encodings are concatenated to the node features.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()

    def forward(
        self,
        data: torch_geometric.data.Data,
    ) -> torch_geometric.data.Data:
        # data["1_cell_curvature"] has shape [num_edges, 1]
        values, indices = data["1_cell_curvature"].sort(dim=0)

        # Find the bounds
        mean_min, mean_max, std_min, std_max = _find_threshold(values.numpy())

        det = std_min + std_max
        lower_bound = torch.tensor(
            (std_min / det) * mean_max + (std_max / det) * mean_min
        )
        upper_bound = torch.tensor(mean_max + std_max)

        values, indices = values.flatten(), indices.flatten()
        # 1. Find edges that hasve to be removed
        # Get top positive curved edges
        most_pos_edges = []
        for val, idx in zip(values, indices, strict=False):
            if val > upper_bound:
                most_pos_edges.append(idx)
            else:
                pass

        # 2. Find edges that have to be added
        # Get top negative curved edges
        most_neg_edges = []
        for val, idx in zip(values, indices, strict=False):
            if val < lower_bound:
                most_neg_edges.append(idx)
            else:
                pass

        # ensure that edges are not repeated
        most_pos_edges = list(set(most_pos_edges))
        most_neg_edges = list(set(most_neg_edges))

        assert (
            len(set(most_pos_edges).intersection(set(most_neg_edges))) == 0
        ), "most_pos_edges and most_neg_edges have intersection"
        # [Dense option]
        incidence_1 = data.incidence_1.to_dense().T
        # Create subset edge index
        subset_edge_index = create_subset_edge_index(
            incidence_1, most_pos_edges
        )

        # # [Sparse option]
        # incidence_1 = data.incidence_1.indices()
        # subset_edge_index = create_subset_edge_index_sparse(incidence_1, most_pos_edges)

        # Assuming the graph is undirected we need to repeat the edges in the opposite direction
        subset_edge_index = torch.cat(
            [subset_edge_index, subset_edge_index.flip(0)], dim=1
        )

        # Remove the edges
        updated_edge_index = remove_edges_preserve_order(
            data.edge_index, subset_edge_index
        )

        # Add the edges to descrease negative curvature
        # [Dense option]
        subset_edge_index = create_subset_edge_index(
            incidence_1, most_neg_edges
        )  # [n_edges, 2]

        # [Sparse option]
        # subset_edge_index = create_subset_edge_index_sparse(incidence_1, most_neg_edges) # [n_edges, 2]
        # Assuming the graph is undirected we need to repeat the edges in the opposite direction
        subset_edge_index = torch.cat(
            [subset_edge_index, subset_edge_index.flip(0)], dim=1
        )

        new_edges = []
        nodes_curvature_w = {}
        assert (
            torch_geometric.utils.is_undirected(subset_edge_index) == True
        ), "subset_edge_index is not undirected, hence might be an error "
        for edge in subset_edge_index.T:
            # check that edge_index is undirected:

            u, v = edge.tolist()

            # The method k_hop_subgraph returns
            # (1) the nodes involved in the subgraph (return direct neighnours)
            # (2) the filtered edge_index connectivity (return subgraph induced by target node)
            # (3) the mapping from node indices in node_idx to their new location
            # (4) the edge mask indicating which edges were preserved.

            num_hops = 1
            neighbourhood_u, edge_index_subgraph_u, _, _ = (
                torch_geometric.utils.k_hop_subgraph(
                    node_idx=u,
                    num_hops=num_hops,
                    edge_index=updated_edge_index,
                    directed=False,
                )
            )
            # neighbourhood_u = torch.arange(data['x'].shape[0])
            neighbourhood_u = neighbourhood_u[
                neighbourhood_u != u
            ]  # Delete the target node itself
            neighbourhood_u = neighbourhood_u[
                neighbourhood_u != v
            ]  # Delete the v node from target node's neighbourhood

            neighbourhood_v, edge_index_subgraph_v, _, _ = (
                torch_geometric.utils.k_hop_subgraph(
                    node_idx=v,
                    num_hops=num_hops,
                    edge_index=updated_edge_index,
                    directed=False,
                )
            )
            neighbourhood_v = neighbourhood_v[
                neighbourhood_v != v
            ]  # Delete the target node itself
            neighbourhood_v = neighbourhood_v[
                neighbourhood_v != u
            ]  # Delete the u node from target node's neighbourhood

            # Get not intersected neightbourhood

            # Construct new edges
            if neighbourhood_u.size(0) > 0:
                w = np.random.choice(neighbourhood_u)
                # Assuming we are constructing undirected graph
                # nodes_curvature_w[w] = data["0_cell_curvature"][w]
                edge = torch.tensor([[v, w], [w, v]], dtype=torch.long)
                new_edges.append(edge)

        new_edges = torch.cat(new_edges).T

        updated_edge_index = torch.cat([updated_edge_index, new_edges], dim=1)

        updated_edge_index = torch_geometric.utils.coalesce(updated_edge_index)
        data.edge_index = updated_edge_index
        # data['nodes_curvature_w'] = nodes_curvature_w

        return data


def create_subset_edge_index_sparse(incidence_1_indices, edges_indices):
    nodes, edges = incidence_1_indices
    subset_edge_index = []
    for target_edge in edges_indices:
        target_edge = target_edge.item()
        node_idx = torch.where(edges == target_edge)[0]
        edge = nodes[node_idx]
        subset_edge_index.append(edge)
    # check that the subset_edge_index is not empty
    if len(subset_edge_index) == 0:
        return torch.tensor([])
    else:
        return torch.stack(subset_edge_index).T


def create_subset_edge_index(incidence_1, edges_indices):
    """Create subset edge index

    Parameters
    ----------
    incidence_1 : torch.Tensor
        The incidence matrix of shape [num_edges, num_nodes]
    edges_indices : torch.Tensor
        The list of indices of the edges to be mapped into the subset edge index format.

    incidence_1 : mapping of edges to nodes, hence every column has exactly two 1s
    edges_indices : [edge_index1, edge_index2, ...]
    edge index format : [[node1, node2, node3, ...], [node2, node3, node4, ...]], hence [node1, node2], [node2, node3],.. fowm edges

    """
    _, nodes = torch.where(incidence_1[edges_indices, :] == 1)
    subset_edge_index = nodes.reshape(len(edges_indices), 2).T
    assert (
        incidence_1[edges_indices, subset_edge_index] == 1
    ).all(), "edge_index is not correct"
    return subset_edge_index


def remove_edges_preserve_order(edge_index, subset_edge_index):
    # Convert subset_edge_index to a set of tuples for fast membership checking
    subset_edge_set = set(map(tuple, subset_edge_index.T.tolist()))

    # Prepare a list to collect the remaining edges
    remaining_edges = []

    # Iterate through the edge_index and exclude edges that are in subset_edge_set
    for edge in edge_index.T.tolist():
        if tuple(edge) not in subset_edge_set:
            remaining_edges.append(edge)

    # Convert the remaining edges back to the edge_index format
    remaining_edge_index = torch.tensor(remaining_edges).T

    assert (
        edge_index.shape[1] - remaining_edge_index.shape[1]
        == subset_edge_index.shape[1]
    ), "The number of edges removed is not correct"

    return remaining_edge_index

    # # Get top negative and positive curved edges
    # most_pos_edges = [edge for edge in curvature_values if afrc.G[edge[0]][edge[1]]['AFRC'] > upper_bound]
    # # most_pos_edges = _C[-batch_remove:]

    # most_neg_edges = [edge for edge in curvature_values if afrc.G[edge[0]][edge[1]]['AFRC'] < lower_bound]
    # # most_neg_edges = _C[:batch_add]

    #     current_iteration += 1
    #     #print(f'Iteration {current_iteration}')

    #     # Remove edges
    #     for (u, v) in most_pos_edges:
    #         if(G.has_edge(u, v)):
    #             G.remove_edge(u, v)

    #     # Add edges
    #     for (u, v) in most_neg_edges:
    #         if list(set(G.neighbors(u)) - set(G.neighbors(v))) != []:
    #             w = np.random.choice(list(set(G.neighbors(u)) - set(G.neighbors(v))))
    #             G.add_edge(v, w)
    #             # add attributes "AFRC", "triangles", and "weight" to each added edge
    #             G[v][w]["AFRC"] = 0.0
    #             G[v][w]["triangles"] = 0
    #             G[v][w]["weight"] = 1.0

    #         elif list(set(G.neighbors(v)) - set(G.neighbors(u))) != []:
    #             w = np.random.choice(list(set(G.neighbors(v)) - set(G.neighbors(u))))
    #             G.add_edge(u, w)
    #             # add attributes "AFRC", "triangles", and "weight" to each added edge
    #             G[u][w]["AFRC"] = 0.0
    #             G[u][w]["triangles"] = 0
    #             G[u][w]["weight"] = 1.0

    #         else:
    #             pass

    #     # except ValueError:
    #     #     continue

    # return data
