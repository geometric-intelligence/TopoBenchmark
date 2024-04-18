### Evaluator for graph classification

from torchmetrics import MetricCollection

from topobenchmarkx.evaluators.metrics import METRICS


class TorchEvaluator:
    r"""Evaluator class that is responsible for computing the metrics for a given task.
    
    Parameters
    ----------
   task : str
        The task type. It can be either "classification" or "regression".
    
    **kwargs : 
        Additional arguments for the class. The arguments depend on the task. 
        In "classification" scenario, the following arguments are expected:
        - num_classes : int
            The number of classes.
        - classification_metrics : list
            A list of classification metrics to be computed.
        
        In "regression" scenario, the following arguments are expected:
        - regression_metrics : list
            A list of regression metrics to be computed.
        
    """
    def __init__(self, task, **kwargs):
        
        # Define the task
        self.task = task        

        # Define the metrics depending on the task
        if kwargs["num_classes"] > 1 and kwargs["task"] == "classification":
            # Note that even for binary classification, we use multiclass metrics
            # Accoding to the torchmetrics documentation (https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html#torchmetrics.classification.MulticlassAccuracy)
            # This setup should work correctly
            parameters = {"num_classes": kwargs["num_classes"]}
            parameters["task"] = "multiclass"
            metric_names = kwargs["classification_metrics"]

        elif kwargs["task"] == "multilabel classification":
            parameters = {"num_classes": kwargs["num_classes"]}
            parameters["task"] = "multilabel"
            metric_names = kwargs["classification_metrics"]

        elif kwargs["task"] == "regression":
            parameters = {}
            metric_names = kwargs["regression_metrics"]

        else:
            raise ValueError(f"Invalid task {kwargs['task']}")

        self.metrics = MetricCollection(
            {name: METRICS[name](**parameters) for name in metric_names}
        )

        self.best_metric = {}

    def update(self, model_out: dict):
        """Update the metrics with the model output.
        
        Parameters
        ----------
        model_out : dict
            The model output. It should contain the following keys:
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

    def compute(self,):
        """Compute the metrics.

        Returns
        -------
        res_dict : dict
            A dictionary containing the computed metrics.
        """

        res_dict = self.metrics.compute()

        return res_dict

    def reset(self,):
        """Reset the metrics. This method should be called after each epoch"""
        self.metrics.reset()


if __name__ == "__main__":
    evaluator = TorchEvaluator(task="classification", num_classes=3, classification_metrics=["accuracy"])
    print(evaluator.task)
