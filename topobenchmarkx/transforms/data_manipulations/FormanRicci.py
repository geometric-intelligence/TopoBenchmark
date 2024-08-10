import networkx as nx




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
        edge_afrc[edge] = _compute_afrc_edge(G, edge[0], edge[1], num_triangles)

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


def _compute_afrc(G: nx.Graph, weight: str="weight") -> nx.Graph:
    """
    Compute Augmented Forman-Ricci curvature for a given NetworkX graph.
    """
    edge_afrc = _compute_afrc_edges(G, weight=weight)

    nx.set_edge_attributes(G, edge_afrc, "AFRC")

    for n in G.nodes():
        afrc_sum = 0
        if G.degree(n) > 1:
            for nbr in G.neighbors(n):
                if 'afrc' in G[n][nbr]:
                    afrc_sum += G[n][nbr]['afrc']

            G.nodes[n]["AFRC"] = afrc_sum / G.degree(n)

    return G 


class FormanRicci:
    """
    A class to compute Forman-Ricci curvature for a given NetworkX graph.
    """

    def __init__(self, G: nx.Graph, weight: str="weight"):
        """
        Initialize a container for Forman-Ricci curvature.
        """
        self.G = G
        self.weight = weight
        self.triangles = []     

        for cycle in _simple_cycles(self.G.to_directed(), 4): # Only compute 3 cycles
            if len(cycle) == 3:
                self.triangles.append(cycle)

        for edge in list(self.G.edges()):
            u, v = edge
            self.G.edges[edge]["triangles"] = len([cycle for cycle in self.triangles if u in cycle and v in cycle])/2


        if not nx.get_edge_attributes(self.G, weight):
            for (v1, v2) in self.G.edges():
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

        # check that all edges have the same attributes
        for edge in self.G.edges():
            if self.G.edges[edge] != edge_attributes:
                edge_attributes = self.G.edges[edge]

                missing_attributes = set(edge_attributes.keys()) - set(self.G.graph.keys())

                if 'weight' in missing_attributes:
                    self.G.edges[edge]['weight'] = 1.0
                    missing_attributes.remove('weight')

                if 'AFRC' in missing_attributes:
                    self.G.edges[edge]['AFRC'] = 0.0
                    missing_attributes.remove('AFRC')

                if 'triangles' in missing_attributes:
                    self.G.edges[edge]['triangles'] = 0.0
                    missing_attributes.remove('triangles')

                assert len(missing_attributes) == 0, 'Missing attributes: %s' % missing_attributes


        return self.G