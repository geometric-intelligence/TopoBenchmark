"""Loss module for the topobenchmarkx package."""

import torch
import torch_geometric

from topobenchmarkx.loss.base import AbstractLoss


class TBXLoss(AbstractLoss):
    r"""Defines the default model loss for the given task.

    Parameters
    ----------
    task : str
        Task type, either "classification" or "regression".
    loss_type : str, optional
        Loss type, either "cross_entropy", "mse", or "mae" (default: None).
    """

    def __init__(self, task, loss_type=None):
        super().__init__()
        self.task = task
        if task == "classification" and loss_type == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss()
        elif task == "classification" and loss_type == "bce":
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif task == "regression" and loss_type == "mse":
            self.criterion = torch.nn.MSELoss()

        elif task == "regression" and loss_type == "mae":
            self.criterion = torch.nn.L1Loss()

        else:
            raise Exception("Loss is not defined")
        self.loss_type = loss_type

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task}, loss_type={self.loss_type})"

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Forward pass of the loss function.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the model output with the loss.
        """
        logits = model_out["logits"]
        target = model_out["labels"]

        if self.task == "regression":
            target = target.unsqueeze(1)

        model_out["loss"] = self.criterion(logits, target)

        return model_out
