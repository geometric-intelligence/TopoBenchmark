"""DataTransform class."""

import inspect

import torch_geometric

from topobenchmark.transforms import LIFTINGS, TRANSFORMS
from topobenchmark.transforms.liftings import (
    GRAPH2CELL_LIFTINGS,
    GRAPH2HYPERGRAPH_LIFTINGS,
    GRAPH2SIMPLICIAL_LIFTINGS,
    Graph2CellLiftingTransform,
    Graph2HypergraphLiftingTransform,
    Graph2SimplicialLiftingTransform,
    LiftingTransform,
)

_map_lifting_types = {
    "graph2cell": (GRAPH2CELL_LIFTINGS, Graph2CellLiftingTransform),
    "graph2hypergraph": (
        GRAPH2HYPERGRAPH_LIFTINGS,
        Graph2HypergraphLiftingTransform,
    ),
    "graph2simplicial": (
        GRAPH2SIMPLICIAL_LIFTINGS,
        Graph2SimplicialLiftingTransform,
    ),
}


def _map_lifting_name(lifting_name):
    for liftings_dict, Transform in _map_lifting_types.values():
        if lifting_name in liftings_dict:
            return Transform

    return LiftingTransform


def _route_lifting_kwargs(kwargs, LiftingMap, Transform):
    lifting_map_sign = inspect.signature(LiftingMap)
    transform_sign = inspect.signature(Transform)

    lifting_map_kwargs = {}
    transform_kwargs = {}

    for key, value in kwargs.items():
        if key in lifting_map_sign.parameters:
            lifting_map_kwargs[key] = value
        elif key in transform_sign.parameters:
            transform_kwargs[key] = value

    return lifting_map_kwargs, transform_kwargs


class DataTransform(torch_geometric.transforms.BaseTransform):
    r"""Abstract class to define a custom data lifting.

    Parameters
    ----------
    transform_name : str
        The name of the transform to be used.
    **kwargs : dict
        Additional arguments for the class. Should contain "transform_name".
    """

    def __init__(self, transform_name, **kwargs):
        super().__init__()

        if transform_name not in LIFTINGS:
            kwargs["transform_name"] = transform_name
            transform = TRANSFORMS[transform_name](**kwargs)
        else:
            LiftingMap_ = TRANSFORMS[transform_name]
            Transform = _map_lifting_name(transform_name)
            lifting_map_kwargs, transform_kwargs = _route_lifting_kwargs(
                kwargs, LiftingMap_, Transform
            )

            lifting_map = LiftingMap_(**lifting_map_kwargs)
            transform = Transform(lifting_map, **transform_kwargs)

        self.parameters = kwargs
        self.transform = transform

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Forward pass of the lifting.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data
            The lifted data.
        """
        transformed_data = self.transform(data)
        return transformed_data
