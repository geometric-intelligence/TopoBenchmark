from abc import abstractmethod

import torch
import torch_geometric

from topobenchmarkx.transforms.liftings.graph_lifting import GraphLifting

__all__ = [
    "HypergraphKHopLifting",
    "HypergraphKNearestNeighborsLifting",
]


class Graph2HypergraphLifting(GraphLifting):
    r"""Abstract class for lifting graphs to hypergraphs.

    Args:
        kwargs (optional): Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "graph2hypergraph"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r})"

    @abstractmethod
    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to hypergraph domain.

        Args:
            data (torch_geometric.data.Data): The input data to be lifted.
        Returns:
            dict: The lifted topology.
        """
        raise NotImplementedError


class HypergraphKHopLifting(Graph2HypergraphLifting):
    r"""Lifts graphs to hypergraph domain by considering k-hop neighborhoods of a node. This lifting extracts a number of hyperedges equal to the number of nodes in the graph.
    
    Args:
        k_value (int, optional): The number of hops to consider. (default: 1)
        kwargs (optional): Additional arguments for the class.
    """
    def __init__(self, k_value=1, **kwargs):
        super().__init__(**kwargs)
        self.k = k_value
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k!r})"

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to hypergraph domain by considering
        k-hop neighborhoods.

        Args:
            data (torch_geometric.data.Data): The input data to be lifted.
        Returns:
            dict: The lifted topology.
        """
        # Check if data has instance x:
        if hasattr(data, "x") and data.x is not None:
            num_nodes = data.x.shape[0]
        else:
            num_nodes = data.num_nodes

        incidence_1 = torch.zeros(num_nodes, num_nodes)
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)

        # Detect isolated nodes
        isolated_nodes = [
            i for i in range(num_nodes) if i not in edge_index[0]
        ]
        if len(isolated_nodes) > 0:
            # Add completely isolated nodes to the edge_index
            edge_index = torch.cat(
                [
                    edge_index,
                    torch.tensor(
                        [isolated_nodes, isolated_nodes], dtype=torch.long
                    ),
                ],
                dim=1,
            )

        for n in range(num_nodes):
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(
                n, self.k, edge_index
            )
            incidence_1[n, neighbors] = 1

        num_hyperedges = incidence_1.shape[1]
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        return {
            "incidence_hyperedges": incidence_1,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }


class HypergraphKNearestNeighborsLifting(Graph2HypergraphLifting):
    r"""Lifts graphs to hypergraph domain by considering k-nearest neighbors. This lifting extracts a number of hyperedges equal to the number of nodes in the graph. The hyperedges all contain the same number of nodes, which is equal to the number of nearest neighbors considered.
    
    Args:
        k_value (int, optional): The number of nearest neighbors to consider. (default: 1)
        loop (bool, optional): If True the hyperedges will contain the node they were created from. (default: True)
        cosine (bool, optional): If True the cosine distance will be used instead of the Euclidean distance. (default: False)
        kwargs (optional): Additional arguments for the class.
    """
    def __init__(self, k_value=1, loop=True, cosine=False, **kwargs):
        super().__init__()
        self.k = k_value
        self.loop = loop
        self.transform = torch_geometric.transforms.KNNGraph(self.k, self.loop, cosine=cosine)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k!r}, loop={self.loop!r})"
    
    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to hypergraph domain by considering
        k-nearest neighbors.
        
        Args:
            data (torch_geometric.data.Data): The input data to be lifted.
        Returns:
            dict: The lifted topology.
        """
        num_nodes = data.x.shape[0]
        data.pos = data.x
        num_hyperedges = num_nodes
        incidence_1 = torch.zeros(num_nodes, num_nodes)
        data_lifted = self.transform(data)
        incidence_1[data_lifted.edge_index[1], data_lifted.edge_index[0]] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        return {
            "incidence_hyperedges": incidence_1,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }
