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
]


__version__ = "0.0.1"
