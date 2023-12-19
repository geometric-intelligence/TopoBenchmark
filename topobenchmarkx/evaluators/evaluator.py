import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


### Evaluator for graph classification
class Evaluator:
    def __init__(self, metrics=[]):
        accepted_metrics = ["rocauc", "acc", "pre", "rec", "f1"]

        if len(metrics) == 0:
            metrics = ["acc", "pre", "rec", "f1"]  # accepted_metrics
        for m in metrics:
            if m not in accepted_metrics:
                raise ValueError(
                    f"Metric {m} is not valid. Choose betweeen {accepted_metrics}."
                )
        self.metrics = metrics

    @property
    def expected_input_format(self):
        input_format = f"Expected input format is a numpy.array or a torch.Tensor of shape (samples, n_classes)."
        return input_format

    def _parse_and_check_input(self, input_dict):
        if not "labels" in input_dict:
            raise RuntimeError("Missing key of y_true")
        if not "logits" in input_dict:
            raise RuntimeError("Missing key of y_pred")

        y_true, y_logits = input_dict["labels"], input_dict["logits"]
        y_pred = y_logits.argmax(dim=-1)

        """
            y_true: numpy ndarray or torch tensor of shape (num_graphs/n_nodes, n_classes)
            y_pred: numpy ndarray or torch tensor of shape (num_graphs/n_nodes, n_classes)
        """

        # converting to torch.Tensor to numpy on cpu
        if torch is not None and isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        if torch is not None and isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        ## check type
        if not isinstance(y_true, np.ndarray):
            raise RuntimeError(
                "Arguments to Evaluator need to be either numpy ndarray or torch tensor"
            )

        if not y_true.shape == y_pred.shape:
            raise RuntimeError("Shape of y_true and y_pred must be the same")

        # if not y_true.ndim == 2:
        #     raise RuntimeError(
        #         "y_true and y_pred mush to 2-dim arrray, {}-dim array given".format(
        #             y_true.ndim
        #         )
        #     )

        return y_true, y_pred

    def eval(self, input_dict):
        results = {}
        y_true, y_pred = self._parse_and_check_input(input_dict)
        # Correct it later to allow roc_auc_score
        res_true = y_true  # np.argmax(y_true, axis=1)
        res_pred = y_pred  # np.argmax(y_pred, axis=1)

        for metric in self.metrics:
            if metric == "rocauc":
                results["roc_auc"] = roc_auc_score(y_true, y_pred)
            if metric == "acc":
                results["acc"] = accuracy_score(res_true, res_pred)
            elif metric == "pre":
                results["pre_micro"] = precision_score(
                    res_true, res_pred, average="micro"
                )
                results["pre_macro"] = precision_score(
                    res_true, res_pred, average="macro"
                )
            elif metric == "rec":
                results["rec_micro"] = recall_score(res_true, res_pred, average="micro")
                results["rec_macro"] = recall_score(res_true, res_pred, average="macro")
            elif metric == "f1":
                results["f1_micro"] = f1_score(res_true, res_pred, average="micro")
                results["f1_macro"] = f1_score(res_true, res_pred, average="macro")

        input_dict["metrics"] = results
        return input_dict


if __name__ == "__main__":
    evaluator = Evaluator()
    print(evaluator.expected_input_format)
