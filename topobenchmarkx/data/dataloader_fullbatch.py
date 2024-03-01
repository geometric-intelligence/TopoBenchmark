from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.data import DataLoader as PyGDataLoader
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
    for b in batch:
        values, keys = b[0], b[1]
        data = MyData()
        for key, value in zip(keys, values):
            if is_sparse(value):
                value = value.coalesce()
            data[key] = value
        data_list.append(data)
    return Batch.from_data_list(data_list)


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
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


class DefaultDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:



    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

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
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


# class TorchGeometricBatchDataModule(LightningDataModule):
#     """`LightningDataModule` for the MNIST dataset.

#     The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
#     It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
#     fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
#     while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
#     technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
#     mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

#     A `LightningDataModule` implements 7 key methods:


#     This allows you to share a full dataset without explaining how to download,
#     split, transform and process the data.

#     Read the docs:
#         https://lightning.ai/docs/pytorch/latest/data/datamodule.html
#     """

#     def __init__(
#         self,
#         dataset_train,
#         dataset_val,
#         dataset_test=None,
#         batch_size: int = 64,
#         num_workers: int = 0,
#         pin_memory: bool = False,
#     ) -> None:
#         """Initialize a `MNISTDataModule`.

#         :param data_dir: The data directory. Defaults to `"data/"`.
#         :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
#         :param batch_size: The batch size. Defaults to `64`.
#         :param num_workers: The number of workers. Defaults to `0`.
#         :param pin_memory: Whether to pin memory. Defaults to `False`.
#         """
#         super().__init__()

#         # this line allows to access init params with 'self.hparams' attribute
#         # also ensures init params will be stored in ckpt
#         self.save_hyperparameters(logger=False)

#         self.dataset_train = dataset_train
#         self.dataset_val = dataset_val
#         self.dataset_test = dataset_test
#         self.batch_size = batch_size

#     def train_dataloader(self) -> DataLoader:
#         """Create and return the train dataloader.

#         :return: The train dataloader.
#         """
#         return PyGDataLoader(
#             dataset=self.dataset_train,
#             batch_size=self.batch_size,
#             num_workers=self.hparams.num_workers,
#             pin_memory=self.hparams.pin_memory,
#             shuffle=True,
#         )

#     def val_dataloader(self) -> DataLoader:
#         """Create and return the validation dataloader.

#         :return: The validation dataloader.
#         """
#         return PyGDataLoader(
#             dataset=self.dataset_val,
#             batch_size=self.batch_size,
#             num_workers=self.hparams.num_workers,
#             pin_memory=self.hparams.pin_memory,
#             shuffle=False,
#         )

#     def test_dataloader(self) -> DataLoader:
#         """Create and return the test dataloader.

#         :return: The test dataloader.
#         """
#         if self.dataset_test == None:
#             raise ValueError("There is no test dataloader.")
#         return PyGDataLoader(
#             dataset=self.dataset_test,
#             batch_size=self.batch_size,
#             num_workers=self.hparams.num_workers,
#             pin_memory=self.hparams.pin_memory,
#             shuffle=False,
#         )

#     def teardown(self, stage: Optional[str] = None) -> None:
#         """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
#         `trainer.test()`, and `trainer.predict()`.

#         :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
#             Defaults to ``None``.
#         """
#         pass

#     def state_dict(self) -> Dict[Any, Any]:
#         """Called when saving a checkpoint. Implement to generate and save the datamodule state.

#         :return: A dictionary containing the datamodule state that you want to save.
#         """
#         return {}

#     def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
#         """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
#         `state_dict()`.

#         :param state_dict: The datamodule state returned by `self.state_dict()`.
#         """
#         pass
