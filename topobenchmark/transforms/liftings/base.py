"""Abstract class for topological liftings."""

import abc

import torch_geometric

from topobenchmark.data.utils import IdentityAdapter
from topobenchmark.transforms.feature_liftings.identity import Identity


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

    def __init__(
        self,
        lifting,
        data2domain=None,
        domain2dict=None,
        domain2domain=None,
        feature_lifting=None,
    ):
        if feature_lifting is None:
            feature_lifting = Identity()

        if data2domain is None:
            data2domain = IdentityAdapter()

        if domain2dict is None:
            domain2dict = IdentityAdapter()

        if domain2domain is None:
            domain2domain = IdentityAdapter()

        if isinstance(lifting, str):
            from topobenchmark.transforms import TRANSFORMS

            lifting = TRANSFORMS[lifting]()

        if isinstance(feature_lifting, str):
            from topobenchmark.transforms import TRANSFORMS

            feature_lifting = TRANSFORMS[feature_lifting]()

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
