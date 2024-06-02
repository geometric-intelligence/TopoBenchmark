import torch
import torch_geometric

from topobenchmarkx.transforms.liftings.graph2hypergraph import (
    Graph2HypergraphLifting,
)


class HypergraphKNNLifting(Graph2HypergraphLifting):
    r"""Lifts graphs to hypergraph domain by considering k-nearest neighbors.

    Args:
        k_value (int, optional): The number of nearest neighbors to consider. (default: 1)
        loop (bool, optional): If True the hyperedges will contain the node they were created from.
        kwargs (optional): Additional arguments for the class.
    """

    def __init__(self, k_value=1, loop=True, **kwargs):
        super().__init__(**kwargs)
        self.k = k_value
        self.loop = loop
        self.transform = torch_geometric.transforms.KNNGraph(self.k, self.loop)

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
        # check for loops, since KNNGraph is inconsistent with nodes with equal features
        if self.loop:
            for i in range(num_nodes):
                if not torch.any(
                    torch.all(
                        data_lifted.edge_index == torch.tensor([[i, i]]).T,
                        dim=0,
                    )
                ):
                    connected_nodes = data_lifted.edge_index[
                        0, data_lifted.edge_index[1] == i
                    ]
                    dists = torch.sqrt(
                        torch.sum(
                            (
                                data.pos[connected_nodes]
                                - data.pos[i].unsqueeze(0) ** 2
                            ),
                            dim=1,
                        )
                    )
                    furthest = torch.argmax(dists)
                    idx = torch.where(
                        torch.all(
                            data_lifted.edge_index
                            == torch.tensor(
                                [[connected_nodes[furthest], i]]
                            ).T,
                            dim=0,
                        )
                    )[0]
                    data_lifted.edge_index[:, idx] = torch.tensor([[i, i]]).T

        incidence_1[data_lifted.edge_index[1], data_lifted.edge_index[0]] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        return {
            "incidence_hyperedges": incidence_1,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }
