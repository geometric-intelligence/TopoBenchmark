import hydra
import torch
from omegaconf import DictConfig


class DefaultLoss:
    """Abstract class that provides an interface to loss logic within netowrk"""

    def __init__(self, task):
        if task == "classification":
            self.criterion = torch.nn.CrossEntropyLoss()

        elif task == "regression":
            self.criterion == torch.nn.mse()

        else:
            raise Exception("Loss is not defined")

    def __call__(self, model_output):
        """Loss logic based on model_output"""

        logits = model_output["logits"]
        target = model_output["labels"]
        model_output["loss"] = self.criterion(logits, target)

        return model_output


class NodeTaskLoss:
    """Abstract class that provides an interface to loss logic within netowrk"""

    def __init__(self, task):
        if task == "classification":
            self.criterion = torch.nn.CrossEntropyLoss()

        elif task == "regression":
            self.criterion == torch.nn.mse()

        else:
            raise Exception("Loss is not defined")

    def __call__(self, model_output):
        """Loss logic based on model_output"""

        logits = model_output["logits"]
        target = model_output["labels"]
        model_output["loss"] = self.criterion(logits, target)

        return model_output


# from abc import ABC, abstractmethod

# import hydra
# from omegaconf import DictConfig

# # logger = logging.getLogger(__name__)


# class AbstractLoss(ABC):
#     """Abstract class that provides an interface to loss logic within netowrk"""

#     def __init__(self, cfg: DictConfig):
#         self.cfg = cfg

#     @abstractmethod
#     def init_loss(
#         self,
#     ):
#         """Initialize loss"""

#     @abstractmethod
#     def forward(self, model_output):
#         """Loss logic based on model_output"""
