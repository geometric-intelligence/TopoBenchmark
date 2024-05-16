from collections import defaultdict
from typing import Any

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.utils import is_sparse
from torch_sparse import SparseTensor


class DomainData(Data):
    r"""Data object class that overwrites some methods from
    `torch_geometric.data.Data` so that not only sparse matrices with adj in the
    name can work with the `torch_geometric` dataloaders."""
    
    def is_valid(self, string):
        r"""Check if the string contains any of the valid names."""
        valid_names = ["adj", "incidence", "laplacian"]
        return any(name in string for name in valid_names)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        r"""Overwrite the `__cat_dim__` method to handle sparse matrices to handle the names specified in `is_valid`."""
        if is_sparse(value) and self.is_valid(key):
            return (0, 1)
        elif "index" in key or key == "face":
            return -1
        else:
            return 0


def to_data_list(batch):
    """Workaround needed since `torch_geometric` doesn't work when using `torch.sparse` instead of `torch_sparse`."""
    for key in batch.keys():
        if batch[key].is_sparse:
            sparse_data = batch[key].coalesce()
            batch[key] = SparseTensor.from_torch_sparse_coo_tensor(sparse_data)
    data_list = batch.to_data_list()
    for i, data in enumerate(data_list):
        for key, d in data:
            if isinstance(data[key], SparseTensor):
                data_list[i][key] = d.to_torch_sparse_coo_tensor()
    return data_list


def collate_fn(batch):
    r"""This function overwrites the `torch_geometric.data.DataLoader` collate function to use the `DomainData` class. This ensures that the `torch_geometric` dataloaders work with sparse matrices that are not necessarily named `adj`. The function also generates the batch slices for the different cell dimensions.
    
    Args:
        batch (list): List of data objects (e.g., `torch_geometric.data.Data`).

    Returns:
        torch_geometric.data.Batch: A `torch_geometric.data.Batch` object.
    """
    data_list = []
    batch_idx_dict = defaultdict(list)

    # Keep track of the running index for each cell dimension
    running_idx = {}

    for batch_idx, b in enumerate(batch):
        values, keys = b[0], b[1]
        data = DomainData()
        for key, value in zip(keys, values, strict=False):
            if is_sparse(value):
                value = value.coalesce()
            data[key] = value

        # Generate batch_slice values for x_1, x_2, x_3, ...
        x_keys = [el for el in keys if ("x_" in el)]
        for x_key in x_keys:
            if x_key != "x_0": 
                if x_key != "x_hyperedges":
                    cell_dim = int(x_key.split("_")[1])
                else:
                    cell_dim = x_key.split("_")[1]

                current_number_of_cells = data[x_key].shape[0]

                batch_idx_dict[f"batch_{cell_dim}"].append(
                    torch.tensor([[batch_idx] * current_number_of_cells])
                )

                if (
                    running_idx.get(f"cell_running_idx_number_{cell_dim}")
                    is None
                ):
                    running_idx[f"cell_running_idx_number_{cell_dim}"] = (
                        current_number_of_cells  
                    )
                
                else:
                    running_idx[f"cell_running_idx_number_{cell_dim}"] += (
                        current_number_of_cells  
                    )

        data_list.append(data)

    batch = Batch.from_data_list(data_list)

    # Rename batch.batch to batch.batch_0 for consistency
    batch["batch_0"] = batch.pop("batch")

    # Add batch slices to batch
    for key, value in batch_idx_dict.items():
        batch[key] = torch.cat(value, dim=1).squeeze(0).long()
    
    # Ensure shape is torch.Tensor
    # "shape" describes the number of n_cells in each graph
    if batch.get("shape") is not None:
        cell_statistics = batch.pop("shape")
        batch["cell_statistics"] = torch.Tensor(cell_statistics).long()

    return batch


class DefaultDataModule(LightningDataModule):
    r"""This class takes care of returning the dataloaders for the training, validation, and test datasets. It also handles the collate function. The class is designed to work with the `torch` dataloaders.

    Args:
        dataset_train (CustomDataset): The training dataset.
        dataset_val (CustomDataset, optional): The validation dataset. (default: None)
        dataset_test (CustomDataset, optional): The test dataset. (default: None)
        batch_size (int, optional): The batch size for the dataloader. (default: 1)
        num_workers (int, optional): The number of worker processes to use for data loading. (default: 0)
        pin_memory (bool, optional): If True, the data loader will copy tensors into pinned memory before returning them. (default: False)

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
        self.save_hyperparameters(
            logger=False,
            ignore=["dataset_train", "dataset_val", "dataset_test"],
        )

        self.dataset_train = dataset_train
        self.batch_size = batch_size

        if dataset_val is None and dataset_test is None:
            # Transductive setting
            self.dataset_val = dataset_train
            self.dataset_test = dataset_train
            assert (
                self.batch_size == 1
            ), "Batch size must be 1 for transductive setting."
        else:
            self.dataset_val = dataset_val
            self.dataset_test = dataset_test
            
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset_train={self.dataset_train}, dataset_val={self.dataset_val}, dataset_test={self.dataset_test}, batch_size={self.batch_size}, num_workers={self.hparams.num_workers}, pin_memory={self.hparams.pin_memory})"

    def train_dataloader(self) -> DataLoader:
        r"""Create and return the train dataloader.

        Returns:
            torch.utils.data.DataLoader: The train dataloader.
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
        r"""Create and return the validation dataloader.

        Returns:
            torch.utils.data.DataLoader: The validation dataloader.
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
        r"""Create and return the test dataloader.

        Returns:
            torch.utils.data.DataLoader: The test dataloader.
        """
        if self.dataset_test is None:
            raise ValueError("There is no test dataloader.")
        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def teardown(self, stage: str | None = None) -> None:
        r"""Lightning hook for cleaning up after `trainer.fit()`,
        `trainer.validate()`, `trainer.test()`, and `trainer.predict()`.

        Args:
            stage (str, optional): The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. (default: None)
        """

    def state_dict(self) -> dict[Any, Any]:
        r"""Called when saving a checkpoint. Implement to generate and save the
        datamodule state.

        Returns:
            dict: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Called when loading a checkpoint. Implement to reload datamodule
        state given datamodule `state_dict()`.

        Args:
            state_dict (dict): The datamodule state. This is the object returned by `state_dict()`.
        """
