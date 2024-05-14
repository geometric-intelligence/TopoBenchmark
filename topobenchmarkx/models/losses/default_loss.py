import torch
import torch_geometric
from topobenchmarkx.models.losses.loss import AbstractltLoss

class DefaultLoss(AbstractltLoss):
    """Abstract class that provides an interface to loss logic within
    netowrk."""

    def __init__(self, task, loss_type=None):
        super().__init__()
        self.task = task
        if task == "classification" and loss_type == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss()

        elif task == "regression" and loss_type == "mse":
            self.criterion = torch.nn.MSELoss()

        elif task == "regression" and loss_type == "mae":
            self.criterion = torch.nn.L1Loss()

        else:
            raise Exception("Loss is not defined")

    def forward(self,  model_out: dict, batch: torch_geometric.data.Data):
        """Loss logic based on model_out."""

        logits = model_out["logits"]
        target = model_out["labels"]

        if self.task == "regression":
            target = target.unsqueeze(1)

        model_out["loss"] = self.criterion(logits, target)

        return model_out
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(task={self.task}, criterion={self.criterion.__class__.__name__})'