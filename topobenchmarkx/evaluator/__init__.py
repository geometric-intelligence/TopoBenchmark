from torchmetrics.classification import AUROC, Accuracy, Precision, Recall
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

# Define metrics
METRICS = {
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "auroc": AUROC,
    "mae": MeanAbsoluteError,
    "mse": MeanSquaredError,
}

from .evaluator import TBXEvaluator  # noqa: E402

__all__ = [
    "TBXEvaluator",
    "METRICS",
]
