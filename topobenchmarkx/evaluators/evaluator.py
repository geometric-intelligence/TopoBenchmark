### Evaluator for graph classification

from torchmetrics import MetricCollection

from topobenchmarkx.evaluators.metrics import METRICS


class TorchEvaluator:
    def __init__(self, **kwargs):
        self.task = kwargs["task"]

        if kwargs["num_classes"] == 1 and kwargs["task"] == "classification":
            parameters = {"num_classes": kwargs["num_classes"]}
            parameters["task"] = "binary"
            metric_names = kwargs["classification_metrics"]

        elif kwargs["num_classes"] > 1 and kwargs["task"] == "classification":
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
        # parameters["task"] = "regression"

        else:
            raise ValueError(f"Invalid task {kwargs['task']}")

        self.metrics = MetricCollection(
            {name: METRICS[name](**parameters) for name in metric_names}
        )

        self.best_metric = {}

    def update(self, model_out: dict):
        preds = model_out["logits"].cpu()
        target = model_out["labels"].cpu()

        if self.task == "regression":
            self.metrics.update(preds, target.unsqueeze(1))

        elif self.task == "classification":
            self.metrics.update(preds, target)

        else:
            raise ValueError(f"Invalid task {self.task}")

    def compute(self, mode):
        res_dict = self.metrics.compute()
        # res_dict = self.update_best_metric(res_dict, mode)

        return res_dict

    # def update_best_metric(self, res_dict, mode):
    #     for key in res_dict:
    #         # Check if "{best_}key" exists
    #         if f"best_{key}" not in self.best_metric:
    #             self.best_metric[f"best_{mode}_{key}"] = res_dict[key]
    #         else:
    #             if res_dict[key] > self.best_metric[f"best_{mode}_{key}"]:
    #                 self.best_metric[f"best_{mode}_{key}"] = res_dict[key]

    #     # Add best metrics to res_dict
    #     res_dict.update(self.best_metric)
    #     return res_dict

    def reset(
        self,
    ):
        self.metrics.reset()


if __name__ == "__main__":
    evaluator = Evaluator()
    print(evaluator.expected_input_format)
