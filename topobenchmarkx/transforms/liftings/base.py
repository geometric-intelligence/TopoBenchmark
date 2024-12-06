"""Abstract class for topological liftings."""

import abc

import torch_geometric

from topobenchmarkx.transforms.converters import IdentityAdapter
from topobenchmarkx.transforms.feature_liftings import FEATURE_LIFTINGS
from topobenchmarkx.transforms.feature_liftings.identity import (
    Identity,
)


class LiftingTransform(torch_geometric.transforms.BaseTransform):
    """Lifting transform.

    Parameters
    ----------
    data2domain : Converter
        Conversion between ``torch_geometric.Data`` into
        domain for consumption by lifting.
    domain2dict : Converter
        Conversion between output domain of feature lifting
        and ``torch_geometric.Data``.
    lifting : LiftingMap
        Lifting map.
    domain2domain : Converter
        Conversion between output domain of lifting
        and input domain for feature lifting.
    feature_lifting : FeatureLiftingMap
        Feature lifting map.
    """

    # NB: emulates previous AbstractLifting
    def __init__(
        self,
        data2domain,
        domain2dict,
        lifting,
        domain2domain=None,
        feature_lifting=None,
    ):
        if feature_lifting is None:
            feature_lifting = Identity()

        if domain2domain is None:
            domain2domain = IdentityAdapter()

        self.data2domain = data2domain
        self.domain2domain = domain2domain
        self.domain2dict = domain2dict
        self.lifting = lifting
        self.feature_lifting = feature_lifting

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Apply the full lifting (topology + features) to the input data.

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

        domain = self.data2domain(data)
        lifted_topology = self.lifting(domain)
        lifted_topology = self.domain2domain(lifted_topology)
        lifted_topology = self.feature_lifting(lifted_topology)
        lifted_topology_dict = self.domain2dict(lifted_topology)

        # TODO: make this line more clear
        return torch_geometric.data.Data(
            **initial_data, **lifted_topology_dict
        )


class LiftingMap(abc.ABC):
    """Lifting map.

    Lifts a domain into another.
    """

    def __call__(self, domain):
        """Lift domain."""
        return self.lift(domain)

    @abc.abstractmethod
    def lift(self, domain):
        """Lift domain."""


class AbstractLifting(torch_geometric.transforms.BaseTransform):
    r"""Abstract class for topological liftings.

    Parameters
    ----------
    feature_lifting : str, optional
        The feature lifting method to be used. Default is 'ProjectionSum'.
    **kwargs : optional
        Additional arguments for the class.
    """

    # TODO: delete

    def __init__(self, feature_lifting=None, **kwargs):
        super().__init__()
        self.feature_lifting = FEATURE_LIFTINGS[feature_lifting]()
        self.neighborhoods = kwargs.get("neighborhoods")

    @abc.abstractmethod
    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift the topology of a graph to higher-order topological domains.

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

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Apply the full lifting (topology + features) to the input data.

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
