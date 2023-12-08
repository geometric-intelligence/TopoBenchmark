import hydra
import torch
from omegaconf import DictConfig

from topobenchmarkx.models.losses.loss import AbstractLoss


class CategoricalCE(AbstractLoss):
    """Abstract class that provides an interface to loss logic within netowrk"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def init_loss(
        self,
    ):
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, model_output):
        """Loss logic based on model_output"""

        logits = model_output["logits"]
        target = model_output["labels"]
        model_output["loss"] = self.criterion(logits, target)

        return model_output
