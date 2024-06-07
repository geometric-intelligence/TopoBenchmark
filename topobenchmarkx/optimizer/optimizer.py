"""Optimizer class responsible of managing both optimizer and scheduler."""

from .base import AbstractOptimizer


class TBXOptimizer(AbstractOptimizer):
    """Optimizer class that manage both optimizer and scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to be used.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Scheduler to be used. Default is None.
    """

    def __init__(self, optimizer, scheduler=None) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_optimizer(self, model_parameters):
        """Configure the optimizer and scheduler.

        Act as a wrapper .

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
