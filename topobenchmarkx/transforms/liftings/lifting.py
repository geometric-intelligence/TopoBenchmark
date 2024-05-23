from abc import abstractmethod

import networkx as nx
import torch_geometric
from torch_geometric.utils.undirected import is_undirected, to_undirected

# Data manipulation transforms
from topobenchmarkx.transforms.data_manipulations import (
    IdentityTransform,
)

# Feature liftings
from topobenchmarkx.transforms.feature_liftings import (
    ConcatentionLifting,
    ProjectionSum,
    SetLifting,
)

FEATURE_LIFTINGS = {
    "projection": ProjectionSum,
    "concatenation": ConcatentionLifting,
    "set": SetLifting,
    None: IdentityTransform,
}


class AbstractLifting(torch_geometric.transforms.BaseTransform):
    r"""Abstract class for topological liftings.

    Parameters
    ----------
    feature_lifting : str, optional
        The feature lifting method to be used. Default is 'projection'.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, feature_lifting=None, **kwargs):
        super().__init__()
        self.feature_lifting = FEATURE_LIFTINGS[feature_lifting]()

    @abstractmethod
    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to higher-order topological domains.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        raise NotImplementedError

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        r"""Applies the full lifting (topology + features) to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data
            The lifted data.
        """
        initial_data = data.to_dict()
        lifted_topology = self.lift_topology(data)
        lifted_topology = self.feature_lifting(lifted_topology)
        return torch_geometric.data.Data(**initial_data, **lifted_topology)


class GraphLifting(AbstractLifting):
    r"""Abstract class for lifting graph topologies to other (topological) domains.

    Parameters
    ----------
    feature_lifting : str, optional
        The feature lifting method to be used. Default is 'projection'.
    preserve_edge_attr : bool, optional
        Whether to preserve edge attributes. Default is False.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(
        self, feature_lifting="ProjectionSum", preserve_edge_attr=False, **kwargs
    ):
        super().__init__(feature_lifting=feature_lifting, **kwargs)
        self.preserve_edge_attr = preserve_edge_attr

    def _data_has_edge_attr(self, data: torch_geometric.data.Data) -> bool:
        r"""Checks if the input data object has edge attributes.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        bool
            Whether the data object has edge attributes.
        """
        return hasattr(data, "edge_attr") and data.edge_attr is not None

    def _generate_graph_from_data(self, data: torch_geometric.data.Data) -> nx.Graph:
        r"""Generates a NetworkX graph from the input data object.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        nx.Graph
            The generated NetworkX graph.
        """
        # Check if data object have edge_attr, return list of tuples as [(node_id, {'features':data}, 'dim':1)] or ??
        nodes = [(n, dict(features=data.x[n], dim=0)) for n in range(data.x.shape[0])]

        if self.preserve_edge_attr and self._data_has_edge_attr(data):
            # In case edge features are given, assign features to every edge
            edge_index, edge_attr = (
                data.edge_index,
                data.edge_attr
                if is_undirected(data.edge_index, data.edge_attr)
                else to_undirected(data.edge_index, data.edge_attr),
            )
            edges = [
                (i.item(), j.item(), dict(features=edge_attr[edge_idx], dim=1))
                for edge_idx, (i, j) in enumerate(
                    zip(edge_index[0], edge_index[1], strict=False)
                )
            ]
            self.contains_edge_attr = True
        else:
            # If edge_attr is not present, return list list of edges
            edges = [
                (i.item(), j.item(), {})
                for i, j in zip(data.edge_index[0], data.edge_index[1], strict=False)
            ]
            self.contains_edge_attr = False
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph


class PointCloudLifting(AbstractLifting):
    r"""Abstract class for lifting point cloud topologies to other (topological) domains.

    Parameters
    ----------
    feature_lifting : str, optional
        The feature lifting method to be used. Default is 'ProjectionSum'.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, feature_lifting="ProjectionSum", **kwargs):
        super().__init__(feature_lifting=feature_lifting, **kwargs)


class CellComplexLifting(AbstractLifting):
    r"""Abstract class for lifting cell complexes to other (topological) domains.

    Parameters
    ----------
    feature_lifting : str, optional
        The feature lifting method to be used. Default is 'ProjectionSum'.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, feature_lifting="ProjectionSum", **kwargs):
        super().__init__(feature_lifting=feature_lifting, **kwargs)


class SimplicialLifting(AbstractLifting):
    r"""Abstract class for lifting simplicial complexes to other (topological) domains.

    Parameters
    ----------
    feature_lifting : str, optional
        The feature lifting method to be used. Default is 'ProjectionSum'.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, feature_lifting="ProjectionSum", **kwargs):
        super().__init__(feature_lifting=feature_lifting, **kwargs)


class HypergraphLifting(AbstractLifting):
    r"""Abstract class for lifting hypergraphs to other (topological) domains.

    Parameters
    ----------
    feature_lifting : str, optional
        The feature lifting method to be used. Default is 'projection'.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, feature_lifting="ProjectionSum", **kwargs):
        super().__init__(feature_lifting=feature_lifting, **kwargs)


class CombinatorialLifting(AbstractLifting):
    r"""Abstract class for lifting combinatorial structures to other (topological) domains.

    Parameters
    ----------
    feature_lifting : str, optional
        The feature lifting method to be used. Default is 'ProjectionSum'.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, feature_lifting="ProjectionSum", **kwargs):
        super().__init__(feature_lifting=feature_lifting, **kwargs)
