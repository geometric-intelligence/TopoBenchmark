### Evaluator for graph classification

from torchmetrics import MetricCollection

from topobenchmarkx.evaluator import METRICS


class TBXEvaluator:
    r"""Evaluator class that is responsible for computing the metrics for a
    given task.

    Args:
        task (str): The task type. It can be either "classification" or "regression".
        **kwargs : Additional arguments for the class. The arguments depend on the task.
            In "classification" scenario, the following arguments are expected:
            - num_classes (int): The number of classes.
            - classification_metrics (list[str]): A list of classification metrics to be computed.

            In "regression" scenario, the following arguments are expected:
            - regression_metrics (list[str]): A list of regression metrics to be computed.
    """

    def __init__(self, task, **kwargs):
        # Define the task
        self.task = task

        # Define the metrics depending on the task
        if kwargs["num_classes"] > 1 and self.task == "classification":
            # Note that even for binary classification, we use multiclass metrics
            # Accoding to the torchmetrics documentation (https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html#torchmetrics.classification.MulticlassAccuracy)
            # This setup should work correctly
            parameters = {"num_classes": kwargs["num_classes"]}
            parameters["task"] = "multiclass"
            metric_names = kwargs["classification_metrics"]

        elif self.task == "multilabel classification":
            parameters = {"num_classes": kwargs["num_classes"]}
            parameters["task"] = "multilabel"
            metric_names = kwargs["classification_metrics"]

        elif self.task == "regression":
            parameters = {}
            metric_names = kwargs["regression_metrics"]

        else:
            raise ValueError(f"Invalid task {kwargs['task']}")

        # self.metrics = MetricCollection(
        #     {name: METRICS[name](**parameters) for name in metric_names}
        # )

        metrics = {}
        for name in metric_names:
            if name in ["recall", "precision", "auroc"]:
                metrics[name] = METRICS[name](average="macro", **parameters)

            else:
                metrics[name] = METRICS[name](**parameters)
        self.metrics = MetricCollection(metrics)

        self.best_metric = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task}, metrics={self.metrics})"
        
    def update(self, model_out: dict):
        r"""Update the metrics with the model output.

        Args:
            model_out (dict): The model output. It should contain the following keys:
                - logits : torch.Tensor
                    The model predictions.
                - labels : torch.Tensor
                    The ground truth labels.
        """
        preds = model_out["logits"].cpu()
        target = model_out["labels"].cpu()

        if self.task == "regression":
            self.metrics.update(preds, target.unsqueeze(1))

        elif self.task == "classification":
            self.metrics.update(preds, target)

        else:
            raise ValueError(f"Invalid task {self.task}")

    def compute(self):
        r"""Compute the metrics.

        Args:
            res_dict (dict): A dictionary containing the computed metrics.
        """
        return self.metrics.compute()

    def reset(self):
        """Reset the metrics.

        This method should be called after each epoch
        """
        self.metrics.reset()
