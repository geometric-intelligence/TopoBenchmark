import abc

import networkx as nx
import numpy as np
import torch
import torch_geometric
from topomodelx.utils.sparse import from_sparse
from torch_geometric.utils.undirected import is_undirected, to_undirected

from topobenchmarkx.complex import PlainComplex
from topobenchmarkx.data.utils.utils import (
    generate_zero_sparse_connectivity,
    select_neighborhoods_of_interest,
)


class Converter(abc.ABC):
    """Convert between data structures representing the same domain."""

    def __call__(self, domain):
        """Convert domain's data structure."""
        return self.convert(domain)

    @abc.abstractmethod
    def convert(self, domain):
        """Convert domain's data structure."""


class IdentityConverter(Converter):
    """Identity conversion.

    Retrieves same data structure for domain.
    """

    def convert(self, domain):
        """Convert domain."""
        return domain


class Data2NxGraph(Converter):
    """Data to nx.Graph conversion.

    Parameters
    ----------
    preserve_edge_attr : bool
        Whether to preserve edge attributes.
    """

    def __init__(self, preserve_edge_attr=False):
        self.preserve_edge_attr = preserve_edge_attr

    def _data_has_edge_attr(self, data: torch_geometric.data.Data) -> bool:
        r"""Check if the input data object has edge attributes.

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

    def convert(self, domain: torch_geometric.data.Data) -> nx.Graph:
        r"""Generate a NetworkX graph from the input data object.

        Parameters
        ----------
        domain : torch_geometric.data.Data
            The input data.

        Returns
        -------
        nx.Graph
            The generated NetworkX graph.
        """
        # Check if data object have edge_attr, return list of tuples as [(node_id, {'features':data}, 'dim':1)] or ??
        nodes = [
            (n, dict(features=domain.x[n], dim=0))
            for n in range(domain.x.shape[0])
        ]

        if self.preserve_edge_attr and self._data_has_edge_attr(domain):
            # In case edge features are given, assign features to every edge
            edge_index, edge_attr = (
                domain.edge_index,
                (
                    domain.edge_attr
                    if is_undirected(domain.edge_index, domain.edge_attr)
                    else to_undirected(domain.edge_index, domain.edge_attr)
                ),
            )
            edges = [
                (i.item(), j.item(), dict(features=edge_attr[edge_idx], dim=1))
                for edge_idx, (i, j) in enumerate(
                    zip(edge_index[0], edge_index[1], strict=False)
                )
            ]

        else:
            # If edge_attr is not present, return list list of edges
            edges = [
                (i.item(), j.item(), {})
                for i, j in zip(
                    domain.edge_index[0], domain.edge_index[1], strict=False
                )
            ]
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph


class Complex2PlainComplex(Converter):
    """toponetx.Complex to PlainComplex conversion.

    NB: order of features plays a crucial role, as ``PlainComplex``
    simply stores them as lists (i.e. the reference to the indices
    of the simplex are lost).

    Parameters
    ----------
    max_rank : int
        Maximum rank of the complex.
    neighborhoods : list, optional
        List of neighborhoods of interest.
    signed : bool, optional
        If True, returns signed connectivity matrices.
    transfer_features : bool, optional
        Whether to transfer features.
    """

    def __init__(
        self,
        max_rank=None,
        neighborhoods=None,
        signed=False,
        transfer_features=True,
    ):
        super().__init__()
        self.max_rank = max_rank
        self.neighborhoods = neighborhoods
        self.signed = signed
        self.transfer_features = transfer_features

    def convert(self, domain):
        """Convert toponetx.Complex to PlainComplex.

        Parameters
        ----------
        domain : toponetx.Complex

        Returns
        -------
        PlainComplex
        """
        # NB: just a slightly rewriting of get_complex_connectivity

        max_rank = self.max_rank or domain.dim
        signed = self.signed
        neighborhoods = self.neighborhoods

        connectivity_infos = [
            "incidence",
            "down_laplacian",
            "up_laplacian",
            "adjacency",
            "coadjacency",
            "hodge_laplacian",
        ]

        practical_shape = list(
            np.pad(list(domain.shape), (0, max_rank + 1 - len(domain.shape)))
        )
        data = {
            connectivity_info: [] for connectivity_info in connectivity_infos
        }
        for rank_idx in range(max_rank + 1):
            for connectivity_info in connectivity_infos:
                try:
                    data[connectivity_info].append(
                        from_sparse(
                            getattr(domain, f"{connectivity_info}_matrix")(
                                rank=rank_idx, signed=signed
                            )
                        )
                    )
                except ValueError:
                    if connectivity_info == "incidence":
                        data[connectivity_info].append(
                            generate_zero_sparse_connectivity(
                                m=practical_shape[rank_idx - 1],
                                n=practical_shape[rank_idx],
                            )
                        )
                    else:
                        data[connectivity_info].append(
                            generate_zero_sparse_connectivity(
                                m=practical_shape[rank_idx],
                                n=practical_shape[rank_idx],
                            )
                        )

        # TODO: handle this
        if neighborhoods is not None:
            data = select_neighborhoods_of_interest(data, neighborhoods)

        # TODO: simplex specific?
        # TODO: how to do this for other?
        if self.transfer_features and hasattr(
            domain, "get_simplex_attributes"
        ):
            # TODO: confirm features are in the right order; update this
            data["features"] = []
            for rank in range(max_rank + 1):
                rank_features_dict = domain.get_simplex_attributes(
                    "features", rank
                )
                if rank_features_dict:
                    rank_features = torch.stack(
                        list(rank_features_dict.values())
                    )
                else:
                    rank_features = None
                data["features"].append(rank_features)

        return PlainComplex(**data)


class PlainComplex2Dict(Converter):
    """PlainComplex to dict conversion."""

    def convert(self, domain):
        """Convert PlainComplex to dict.

        Parameters
        ----------
        domain : toponetx.Complex

        Returns
        -------
        dict
        """
        data = {}
        connectivity_infos = [
            "incidence",
            "down_laplacian",
            "up_laplacian",
            "adjacency",
            "coadjacency",
            "hodge_laplacian",
        ]
        for connectivity_info in connectivity_infos:
            info = getattr(domain, connectivity_info)
            for rank, rank_info in enumerate(info):
                data[f"{connectivity_info}_{rank}"] = rank_info

        # TODO: handle neighborhoods
        data["shape"] = domain.shape

        for index, values in enumerate(domain.features):
            if values is not None:
                data[f"x_{index}"] = values

        return data


class ConverterComposition(Converter):
    def __init__(self, converters):
        super().__init__()
        self.converters = converters

    def convert(self, domain):
        """Convert domain"""
        for converter in self.converters:
            domain = converter(domain)

        return domain


class Complex2Dict(ConverterComposition):
    """Complex to dict conversion.

    Parameters
    ----------
    max_rank : int
        Maximum rank of the complex.
    neighborhoods : list, optional
        List of neighborhoods of interest.
    signed : bool, optional
        If True, returns signed connectivity matrices.
    transfer_features : bool, optional
        Whether to transfer features.
    """

    def __init__(
        self,
        max_rank=None,
        neighborhoods=None,
        signed=False,
        transfer_features=True,
    ):
        complex2plain = Complex2PlainComplex(
            max_rank=max_rank,
            neighborhoods=neighborhoods,
            signed=signed,
            transfer_features=transfer_features,
        )
        plain2dict = PlainComplex2Dict()
        super().__init__(converters=(complex2plain, plain2dict))
