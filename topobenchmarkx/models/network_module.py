from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from topobenchmarkx.models.wrappers.default_wrapper import HypergraphWrapper, SANWrapper

# import topomodelx


class NetworkModule(LightningModule):
    """A `LightningModule` implements 8 key methods:

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        readout: torch.nn.Module,
        loss: torch.nn.Module,
        backbone_wrapper: torch.nn.Module,
        feature_encoder: Union[torch.nn.Module, None] = None,
        **kwargs
        # optimizer: torch.optim.Optimizer,
        # scheduler: torch.optim.lr_scheduler,
        # compile: bool,
    ) -> None:
        """Initialize a `NetworkModule`.

        :param backbone: The backbone model to train.
        :param readout: The readout class.
        :param loss: The loss class.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, ignore=["train_dataset", "validation_dataset", "test_dataset"]
        )

        self.feature_encoder = feature_encoder
        self.backbone = backbone_wrapper(backbone)
        self.readout = readout
        self.evaluator = None

        # loss function
        self.task_level = self.hparams["readout"].task_level
        self.criterion = loss

        # Tracking best so far validation accuracy
        self.val_acc_best = MeanMetric()
        self.metric_collector_val = []
        self.metric_collector_val2 = []
        self.metric_collector_test = []

    def forward(self, batch) -> dict:
        """Perform a forward pass through the model `self.backbone`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.backbone(batch)

    def model_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """

        if self.feature_encoder:
            batch = self.feature_encoder(batch)

        model_out = self.forward(batch)
        model_out = self.readout(model_out)

        model_out = self.process_outputs(batch, model_out)

        model_out = self.criterion(model_out)
        self.evaluator.update(model_out)

        # model_out = self.criterion(model_out)

        return model_out

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        self.state_str = "Training"
        model_out = self.model_step(batch)

        # if self.task_level == "node":
        #     # Keep only train data points
        #     for key, val in model_out.items():
        #         if key in ["logits", "labels"]:
        #             model_out[key] = val[batch.train_mask]

        # # Criterion
        # model_out = self.criterion(model_out)

        # # Evaluation
        # self.evaluator.update(model_out)

        # Update and log metrics
        self.log(
            "train/loss",
            model_out["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )

        # Return loss for backpropagation step
        return model_out["loss"]

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        self.state_str = "Validation"
        model_out = self.model_step(batch)

        # # Keep only validation data points
        # if self.task_level == "node":
        #     for key, val in model_out.items():
        #         # if key not in ["loss", "hyperedge"]:
        #         if key in ["logits", "labels"]:
        #             model_out[key] = val[batch.val_mask]

        # # Criterion
        # model_out = self.criterion(model_out)

        # # Evaluation
        # self.evaluator.update(model_out)
        # self.metric_collector_val.append((model_out["logits"], model_out["labels"]))

        # Log Loss
        self.log(
            "val/loss",
            model_out["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        self.state_str = "Test"
        model_out = self.model_step(batch)

        # if self.task_level == "node":
        #     # Keep only test data points
        #     for key, val in model_out.items():
        #         if key in ["logits", "labels"]:
        #             model_out[key] = val[batch.test_mask]

        # # Criterion
        # model_out = self.criterion(model_out)

        # Log loss
        self.log(
            "test/loss",
            model_out["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )

        # Evaluation
        # self.evaluator.update(model_out)
        # self.metric_collector_test.append((model_out["logits"], model_out["labels"]))

    def process_outputs(self, batch, model_out: dict) -> dict:
        """Process model outputs."""

        # Get the correct mask
        if self.state_str == "Training":
            mask = batch.train_mask
        elif self.state_str == "Validation":
            mask = batch.val_mask
        elif self.state_str == "Test":
            mask = batch.test_mask
        else:
            raise ValueError("Invalid state_str")

        if self.task_level == "node":
            # Keep only train data points
            for key, val in model_out.items():
                if key in ["logits", "labels"]:
                    model_out[key] = val[mask]

        return model_out

    def log_metrics(self, mode=None):
        """Log metrics."""
        metrics_dict = self.evaluator.compute()
        for key in metrics_dict.keys():
            self.log(
                f"{mode}/{key}",
                metrics_dict[key],
                prog_bar=True,
                on_step=False,
            )

        # Reset evaluator for next epoch
        self.evaluator.reset()

    def on_validation_epoch_start(self) -> None:
        """According pytorch lightning documentation, this hook is called at the beginning of the validation epoch.

        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks

        Note that the validation step is within the train epoch. Hence here we have to log the train metrics
        before we reset the evaluator to start the validation loop.
        """

        # Log train metrics and reset evaluator
        self.log_metrics(mode="train")

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Log validation metrics and reset evaluator
        self.log_metrics(mode="val")

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.log_metrics(mode="test")
        print()

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.evaluator.reset()

    def on_val_epoch_start(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.evaluator.reset()

    def on_test_epoch_start(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.evaluator.reset()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(
            params=list(self.trainer.model.parameters())
            + list(self.readout.parameters())
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
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


# Collect validation statistics
# self.val_acc_best.update(model_out["metrics"]["acc"])
# self.metric_collector.append(model_out["metrics"]["acc"])


# def on_train_start(self) -> None:
#     """Lightning hook that is called when training begins."""
#     # by default lightning executes validation step sanity checks before training starts,
#     # so it's worth to make sure validation metrics don't store results from these checks
#     # self.val_loss.reset()
#     # self.val_acc.reset()
#     self.val_acc_best.reset()


# def on_validation_epoch_end(self) -> None:
#     "Lightning hook that is called when a validation epoch ends."
#     pass
# self.criterion = torch.nn.CrossEntropyLoss()

# self.evaluator = evaluator
# # metric objects for calculating and averaging accuracy across batches
# self.train_acc = Accuracy(task="multiclass", num_classes=7)
# self.val_acc = Accuracy(task="multiclass", num_classes=7)
# self.test_acc = Accuracy(task="multiclass", num_classes=7)

# for averaging loss across batches
# self.train_loss = MeanMetric()
# self.val_loss = MeanMetric()
# self.test_loss = MeanMetric()

if __name__ == "__main__":
    _ = NetworkModule(None, None, None, None)
