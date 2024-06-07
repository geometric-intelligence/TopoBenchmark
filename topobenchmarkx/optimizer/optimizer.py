"""Optimizer class responsible of managing both optimizer and scheduler."""

import functools

import torch.optim

from .base import AbstractOptimizer

TORCH_OPTIMIZERS = torch.optim.__dict__
TORCH_SCHEDULERS = torch.optim.lr_scheduler.__dict__


class TBXOptimizer(AbstractOptimizer):
    """Optimizer class that manage both optimizer and scheduler.

    Parameters
    ----------
    optimizer_id : torch.optim.Optimizer
        Optimizer to be used.
    parameters : dict
        Parameters to be passed to the optimizer.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Scheduler to be used. Default is None.
    """

    def __init__(self, optimizer_id, parameters, scheduler=None) -> None:
        optimizer_id = optimizer_id
        self.optimizer = functools.partial(
            TORCH_OPTIMIZERS[optimizer_id], **parameters
        )
        if scheduler is not None:
            scheduler_id = scheduler.get("scheduler_id")
            scheduler_params = scheduler.get("scheduler_params")
            self.scheduler = functools.partial(
                TORCH_SCHEDULERS[scheduler_id], **scheduler_params
            )
        else:
            self.scheduler = None

    def __repr__(self) -> str:
        if self.scheduler is not None:
            return f"{self.__class__.__name__}(optimizer={self.optimizer.__name__}, scheduler={self.scheduler.__name__})"
        else:
            return f"{self.__class__.__name__}(optimizer={self.optimizer.__name__})"

    def configure_optimizer(self, model_parameters):
        """Configure the optimizer and scheduler.

        Act as a wrapper to provide Trainer the required config dict.

        Parameters
        ----------
        model_parameters : dict
            The model parameters.

        Returns
        -------
        dict
            The optimizer and scheduler configuration.
        """
        optimizer = self.optimizer(params=model_parameters)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
