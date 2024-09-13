import networkx as nx
import numpy as np
import torch
import torch_geometric
from sklearn.mixture import GaussianMixture
from torch_geometric.utils import (
    from_networkx,
    to_networkx,
)


def _compute_afrc_edge(G: nx.Graph, ni: int, nj: int, t_num: int) -> float:
    """
    Computes the Augmented Forman-Ricci curvature of a given edge
    """
    afrc = 4 - G.degree(ni) - G.degree(nj) + 3 * t_num
    return afrc


def _compute_afrc_edges(G: nx.Graph, weight="weight", edge_list=[]) -> dict:
    """
    Compute Augmented Forman-Ricci curvature for edges in  given edge lists.
    """
    if edge_list == []:
        edge_list = G.edges()

    edge_afrc = {}
    for edge in edge_list:
        num_triangles = G.edges[edge]["triangles"]
        edge_afrc[edge] = _compute_afrc_edge(
            G, edge[0], edge[1], num_triangles
        )

    return edge_afrc


def _simple_cycles(G: nx.Graph, limit: int = 3):
    """
    Find simple cycles (elementary circuits) of a graph up to a given length.
    """
    subG = type(G)(G.edges())
    sccs = list(nx.strongly_connected_components(subG))
    while sccs:
        scc = sccs.pop()
        startnode = scc.pop()
        path = [startnode]
        blocked = set()
        blocked.add(startnode)
        stack = [(startnode, list(subG[startnode]))]

        while stack:
            thisnode, nbrs = stack[-1]

            if nbrs and len(path) < limit:
                nextnode = nbrs.pop()
                if nextnode == startnode:
                    yield path[:]
                elif nextnode not in blocked:
                    path.append(nextnode)
                    stack.append((nextnode, list(subG[nextnode])))
                    blocked.add(nextnode)
                    continue
            if not nbrs or len(path) >= limit:
                blocked.remove(thisnode)
                stack.pop()
                path.pop()
        subG.remove_node(startnode)
        H = subG.subgraph(scc)
        sccs.extend(list(nx.strongly_connected_components(H)))


def _compute_afrc(G: nx.Graph, weight: str = "weight") -> nx.Graph:
    """
    Compute Augmented Forman-Ricci curvature for a given NetworkX graph.
    """
    edge_afrc = _compute_afrc_edges(G, weight=weight)

    nx.set_edge_attributes(G, edge_afrc, "AFRC")

    for n in G.nodes():
        afrc_sum = 0
        if G.degree(n) > 1:
            for nbr in G.neighbors(n):
                if "AFRC" in G[n][nbr]:  # if 'afrc' in G[n][nbr]:
                    afrc_sum += G[n][nbr]["AFRC"]

            G.nodes[n]["AFRC"] = afrc_sum / G.degree(n)

    return G


class FormanRicci:
    """
    A class to compute Forman-Ricci curvature for a given NetworkX graph.
    """

    def __init__(self, G: nx.Graph, weight: str = "weight"):
        """
        Initialize a container for Forman-Ricci curvature.
        """
        self.G = G
        self.weight = weight
        self.triangles = []

        for cycle in _simple_cycles(
            self.G.to_directed(), 4
        ):  # Only compute 3 cycles
            if len(cycle) == 3:
                self.triangles.append(cycle)

        for edge in list(self.G.edges()):
            u, v = edge
            self.G.edges[edge]["triangles"] = (
                len(
                    [
                        cycle
                        for cycle in self.triangles
                        if u in cycle and v in cycle
                    ]
                )
                / 2
            )

        if not nx.get_edge_attributes(self.G, weight):
            for v1, v2 in self.G.edges():
                self.G[v1][v2][weight] = 1.0

        self_loop_edges = list(nx.selfloop_edges(self.G))
        if self_loop_edges:
            self.G.remove_edges_from(self_loop_edges)

    def compute_afrc_edges(self, edge_list=None):
        """
        Compute Augmented Forman-Ricci curvature for edges in  given edge lists.
        """
        if edge_list is None:
            edge_list = self.G.edges()
        else:
            edge_list = list(edge_list)

        return _compute_afrc_edges(self.G, self.weight, edge_list)

    def compute_ricci_curvature(self) -> nx.Graph:
        """
        Compute AFRC of edges and nodes.
        """
        self.G = _compute_afrc(self.G, self.weight)

        edge_attributes = self.G.graph

        # # check that all edges have the same attributes
        # for edge in self.G.edges():
        #     if self.G.edges[edge] != edge_attributes:
        #         edge_attributes = self.G.edges[edge]

        #         missing_attributes = set(edge_attributes.keys()) - set(self.G.graph.keys())

        #         if 'weight' in missing_attributes:
        #             self.G.edges[edge]['weight'] = 1.0
        #             missing_attributes.remove('weight')

        #         if 'AFRC' in missing_attributes:
        #             self.G.edges[edge]['AFRC'] = 0.0
        #             missing_attributes.remove('AFRC')

        #         if 'triangles' in missing_attributes:
        #             self.G.edges[edge]['triangles'] = 0.0
        #             missing_attributes.remove('triangles')

        #         assert len(missing_attributes) == 0, 'Missing attributes: %s' % missing_attributes

        return self.G


def _preprocess_data(data, is_undirected=True):
    # Get necessary data information
    N = data.x.shape[0]
    m = data.edge_index.shape[1]

    # Compute the adjacency matrix
    if "edge_type" not in data.keys():
        edge_type = np.zeros(m, dtype=int)
    else:
        edge_type = data.edge_type

    # Convert graph to Networkx
    G = to_networkx(data)
    if is_undirected:
        G = G.to_undirected()

    return G, N, edge_type


def _find_threshold(curv_vals: np.ndarray, formula_type="correct") -> float:
    """
    Model the curvature distribution with a mixture of two Gaussians.
    Find the midpoint between the means of the two Gaussians.
    """
    gmm = GaussianMixture(n_components=2, random_state=0).fit(curv_vals)

    mean1 = gmm.means_[0][0]
    std1 = np.sqrt(gmm.covariances_[0][0][0])

    mean2 = gmm.means_[1][0]
    std2 = np.sqrt(gmm.covariances_[1][0][0])
    # mean2 * std1 + mean1 * std2
    if formula_type == "correct":
        threshold = (mean2 * std1 + mean1 * std2) / (std1 + std2)
    elif formula_type == "incorrect":
        threshold = (mean1 * std1 + mean2 * std2) / (std1 + std2)
    else:
        raise ValueError("Invalid formula type: %s" % formula_type)

    return (threshold, mean1, std1, mean2, std2)


class RewireAFRC3(torch_geometric.transforms.BaseTransform):
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
        loops: int = 10,
        **kwargs,
    ) -> None:
        self.loops = loops
        self.lower_bound_eq = kwargs["lower_bound_eq"]
        self.compute_every_it = kwargs["compute_every_it"]

    def forward(
        self,
        data: torch_geometric.data.Data,
    ) -> torch_geometric.data.Data:
        init_edges = data.edge_index.clone()
        data.edge_index, _ = self.borf4(data)
        return data

    # afrc-3 based rewiring
    def borf4(self, data):
        # Extractc curvature information from tbx data
        # incidence_1 = data.incidence_1.to_dense().T
        # _, nodes = torch.where(incidence_1[:, :]==1)
        # data_edges = nodes.reshape(-1,2)
        # edge2curvature={tuple(edge.numpy()):data["1_cell_curvature"][idx].item() for idx, edge in enumerate(data_edges)}

        # Preprocess data
        G, N, edge_type = _preprocess_data(data)

        # Rewiring begins
        current_iteration = 0
        lower_bound_statistic = []
        for _ in range(self.loops):
            # try:
            afrc = FormanRicci(G)
            afrc.compute_ricci_curvature()

            # if current_iteration==0:
            #     # Check that the curvature values are correct for first iteration
            #     assert np.array([afrc.G[key[0]][key[1]]['AFRC']==value for key, value in edge2curvature.items()]).all() == True, "Curvature values are not correct"

            _C = sorted(afrc.G.edges, key=lambda x: afrc.G[x[0]][x[1]]["AFRC"])

            curvature_values = [
                afrc.G[edge[0]][edge[1]]["AFRC"] for edge in _C
            ]

            # find the bounds
            if self.compute_every_it == True:
                lower_bound, mean1, std1, mean2, std2 = _find_threshold(
                    np.array(curvature_values).reshape(-1, 1),
                    formula_type=self.lower_bound_eq,
                )
                if mean1 > mean2:
                    upper_bound = mean1 + std1
                else:
                    upper_bound = mean2 + std2

            else:
                if current_iteration == 0:
                    lower_bound, mean1, std1, mean2, std2 = _find_threshold(
                        np.array(curvature_values).reshape(-1, 1),
                        formula_type=self.lower_bound_eq,
                    )
                    if mean1 > mean2:
                        upper_bound = mean1 + std1
                    else:
                        upper_bound = mean2 + std2

            # Get top negative and positive curved edges
            most_pos_edges = [
                edge
                for edge in _C
                if afrc.G[edge[0]][edge[1]]["AFRC"] > upper_bound
            ]
            # most_pos_edges = _C[-batch_remove:]

            most_neg_edges = [
                edge
                for edge in _C
                if afrc.G[edge[0]][edge[1]]["AFRC"] < lower_bound
            ]
            lower_bound_statistic.append(
                {
                    "n_most_negative": len(most_neg_edges),
                    "lower_bound": lower_bound,
                }
            )
            # most_neg_edges = _C[:batch_add]

            current_iteration += 1
            # print(f'Iteration {current_iteration}')

            # Remove edges
            for u, v in most_pos_edges:
                if G.has_edge(u, v):
                    G.remove_edge(u, v)

            # Add edges
            for u, v in most_neg_edges:
                if list(set(G.neighbors(u)) - set(G.neighbors(v))) != []:
                    w = np.random.choice(
                        list(set(G.neighbors(u)) - set(G.neighbors(v)))
                    )
                    G.add_edge(v, w)
                    # add attributes "AFRC", "triangles", and "weight" to each added edge
                    G[v][w]["AFRC"] = 0.0
                    G[v][w]["triangles"] = 0
                    G[v][w]["weight"] = 1.0

                elif list(set(G.neighbors(v)) - set(G.neighbors(u))) != []:
                    w = np.random.choice(
                        list(set(G.neighbors(v)) - set(G.neighbors(u)))
                    )
                    G.add_edge(u, w)
                    # add attributes "AFRC", "triangles", and "weight" to each added edge
                    G[u][w]["AFRC"] = 0.0
                    G[u][w]["triangles"] = 0
                    G[u][w]["weight"] = 1.0

                else:
                    pass

        edge_attributes = G.graph

        problematic_edges = 0

        # check that all edges have the same attributes
        for edge in G.edges():
            if G.edges[edge] != edge_attributes:
                problematic_edges += 1

                edge_attributes = G.edges[edge]

                missing_attributes = set(edge_attributes.keys()) - set(
                    G.graph.keys()
                )

                if "weight" in missing_attributes:
                    G.edges[edge]["weight"] = 1.0
                    missing_attributes.remove("weight")

                if "AFRC" in missing_attributes:
                    G.edges[edge]["AFRC"] = 0.0
                    missing_attributes.remove("AFRC")

                if "triangles" in missing_attributes:
                    G.edges[edge]["triangles"] = 0.0
                    missing_attributes.remove("triangles")

                assert len(missing_attributes) == 0, (
                    "Missing attributes: %s" % missing_attributes
                )

        # print('Number of edges with missing attributes: %d' % problematic_edges)

        for node in G.nodes():
            if "AFRC" not in G.nodes[node]:
                G.nodes[node]["AFRC"] = 0.0

        if G.number_of_nodes() > 0:
            node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())

        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            if set(feat_dict.keys()) != set(node_attrs):
                # raise an error and print the missing attributes
                if set(node_attrs) - set(feat_dict.keys()) != set():
                    missing_node_attributes = set(node_attrs) - set(
                        feat_dict.keys()
                    )
                else:
                    missing_node_attributes = set(feat_dict.keys()) - set(
                        node_attrs
                    )
                raise ValueError(
                    "Node %d is missing attributes %s"
                    % (i, missing_node_attributes)
                )

        edge_index = from_networkx(G).edge_index
        edge_type = torch.zeros(size=(len(G.edges),)).type(torch.LongTensor)
        edge_index = torch_geometric.utils.remove_self_loops(edge_index)[0]
        print(lower_bound_statistic)
        return edge_index, edge_type
