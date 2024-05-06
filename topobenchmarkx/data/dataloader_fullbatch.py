from collections import defaultdict
from typing import Any, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.utils import is_sparse
from torch_sparse import SparseTensor


class MyData(Data):
    def is_valid(self, string):
        valid_names = ["adj", "incidence", "laplacian"]
        for name in valid_names:
            if name in string:
                return True
        return False

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if is_sparse(value) and self.is_valid(key):
            return (0, 1)
        elif "index" in key or key == "face":
            return -1
        else:
            return 0


def to_data_list(batch):
    """
    Workaround needed since torch_geometric doesn't work well with torch.sparse
    """
    for key in batch.keys():
        if batch[key].is_sparse:
            sparse_data = batch[key].coalesce()
            batch[key] = SparseTensor.from_torch_sparse_coo_tensor(sparse_data)
    data_list = batch.to_data_list()
    for i, data in enumerate(data_list):
        for key in data.keys():
            if isinstance(data[key], SparseTensor):
                data_list[i][key] = data[key].to_torch_sparse_coo_tensor()
    return data_list


def collate_fn(batch):
    """
    args:
        batch - list of (tensor, label)

    reutrn:
        xs - a tensor of all examples in 'batch' after padding
        ys - a LongTensor of all labels in batch
    """
    data_list = []
    batch_idx_dict = defaultdict(list)

    # Keep track of the running index for each cell dimension
    running_idx = {}

    for batch_idx, b in enumerate(batch):
        values, keys = b[0], b[1]
        data = MyData()
        for key, value in zip(keys, values, strict=False):
            if is_sparse(value):
                value = value.coalesce()
            data[key] = value

        # Generate batch_slice values for x_2, x_3, ...
        x_keys = [el for el in keys if ("x_" in el)]
        for x_key in x_keys:
            # current_number_of_nodes = data["x_0"].shape[0]

            if x_key != "x_0" and x_key != "x_hyperedges":
                cell_dim = int(x_key.split("_")[1])
                current_number_of_cells = data[x_key].shape[0]

                batch_idx_dict[f"batch_{cell_dim}"].append(
                    torch.tensor([[batch_idx] * current_number_of_cells])
                )

                if running_idx.get(f"cell_running_idx_number_{cell_dim}") is None:
                    running_idx[f"cell_running_idx_number_{cell_dim}"] = (
                        current_number_of_cells  # current_number_of_nodes
                    )
                else:
                    # Make sure the idx is contiguous
                    data[f"x_{cell_dim}"] = (
                        data[f"x_{cell_dim}"]
                        + running_idx[f"cell_running_idx_number_{cell_dim}"]
                    ).long()

                    running_idx[f"cell_running_idx_number_{cell_dim}"] += (
                        current_number_of_cells  # current_number_of_nodes
                    )

            elif x_key == "x_hyperedges":
                cell_dim = x_key.split("_")[1]
                current_number_of_hyperedges = data[x_key].shape[0]

                batch_idx_dict["batch_hyperedges"].append(
                    torch.tensor([[batch_idx] * current_number_of_hyperedges])
                )

                if running_idx.get(f"cell_running_idx_number_{cell_dim}") is None:
                    running_idx[f"cell_running_idx_number_{cell_dim}"] = (
                        current_number_of_hyperedges
                    )
                else:
                    # Make sure the idx is contiguous
                    data[f"x_{cell_dim}"] = (
                        data[f"x_{cell_dim}"]
                        + running_idx[f"cell_running_idx_number_{cell_dim}"]
                    ).long()

                    running_idx[f"cell_running_idx_number_{cell_dim}"] += (
                        current_number_of_hyperedges
                    )
            else:
                # Function Batch.from_data_list creates a running index automatically
                pass

        data_list.append(data)

    batch = Batch.from_data_list(data_list)

    # Add batch slices to batch
    for key, value in batch_idx_dict.items():
        batch[key] = torch.cat(value, dim=1).squeeze(0).long()
    return batch


class FullBatchDataModule(LightningDataModule):
    """

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = dataset
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # persistent_workers=True,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # persistent_workers=True,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # persistent_workers=True,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """


class DefaultDataModule(LightningDataModule):
    """
    Initializes the DefaultDataModule class.

    Args:
        dataset_train: The training dataset.
        dataset_val: The validation dataset (optional).
        dataset_test: The test dataset (optional).
        batch_size: The batch size for the dataloader.
        num_workers: The number of worker processes to use for data loading.
        pin_memory: If True, the data loader will copy tensors into pinned memory before returning them.

    Returns:
        None

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        dataset_train,
        dataset_val=None,
        dataset_test=None,
        batch_size=1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset_train = dataset_train
        self.batch_size = batch_size

        if dataset_val == None and dataset_test == None:
            # Transductive setting
            self.dataset_val = dataset_train
            self.dataset_test = dataset_train
            assert (
                self.batch_size == 1
            ), "Batch size must be 1 for transductive setting."
        else:
            self.dataset_val = dataset_val
            self.dataset_test = dataset_test

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        if self.dataset_test == None:
            raise ValueError("There is no test dataloader.")
        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
