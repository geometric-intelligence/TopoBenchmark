"""TopobenchmarkX: A library for benchmarking of topological models."""

# Import submodules
from . import (
    data,
    dataloader,
    datasets,
    evaluator,
    loss,
    model,
    nn,
    transforms,
    utils,
)
from .run import initialize_hydra

__all__ = [
    "data",
    "evaluator",
    "loss",
    "nn",
    "transforms",
    "utils",
    "dataloader",
    "datasets",
    "model",
    "initialize_hydra",
]


__version__ = "0.0.1"
