import hydra # noqa: F401
import torch
from omegaconf import DictConfig # noqa: F401


class DefaultLoss:
    """Abstract class that provides an interface to loss logic within netowrk"""

    def __init__(self, task):
        if task == "classification":
            self.criterion = torch.nn.CrossEntropyLoss()

        elif task == "regression":
            self.criterion = torch.nn.mse()
        else:
            raise Exception("Loss is not defined")

    def __call__(self, model_output):
        """Loss logic based on model_output"""

        logits = model_output["logits"]
        target = model_output["labels"]
        model_output["loss"] = self.criterion(logits, target)

        return model_output
