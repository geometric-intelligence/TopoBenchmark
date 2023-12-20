from typing import Any, Dict, Optional

from lightning import LightningDataModule

# from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import DataLoader

# import torch_geometric


# class CustomDataset(torch_geometric.data.Dataset):
#     def __init__(self, data_lst):
#         super().__init__()
#         self.data_lst = data_lst

#     def get(self, idx):
#         data = self.data[idx]
#         return data

#     def len(self):
#         return len(self.data)
# class CustomDataset(Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return 1

#     def __getitem__(self, idx):
#         return (
#             self.data.x,
#             self.data.edge_index,
#             self.data.y,
#             self.data.n_x,
#             self.data.num_hyperedges,
#             self.data.num_class,
#             self.data.train_mask,
#             self.data.val_mask,
#             self.data.test_mask,
#         )


# def collate_fn(batch):
#     """
#     args:
#         batch - list of (tensor, label)

#     reutrn:
#         xs - a tensor of all examples in 'batch' after padding
#         ys - a LongTensor of all labels in batch
#     """
#     # Find longest sequence
#     x = batch[0][0]
#     edge_index = batch[0][1]
#     y = batch[0][2]
#     n_x = batch[0][3]
#     num_hyperedges = batch[0][4]
#     num_class = batch[0][5]
#     train_mask = batch[0][6]
#     val_mask = batch[0][7]
#     test_mask = batch[0][8]

#     return Data(
#         x=x,
#         edge_index=edge_index,
#         n_x=n_x,
#         num_hyperedges=num_hyperedges,
#         num_class=num_class,
#         y=y,
#         train_mask=train_mask,
#         val_mask=val_mask,
#         test_mask=test_mask,
#     )


class FullBatchDataModule(LightningDataModule):
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
        dataset,
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
        # self.data_train: Optional[Dataset] = None
        # self.data_val: Optional[Dataset] = None
        # self.data_test: Optional[Dataset] = None

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            # collate_fn=collate_fn,
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
            shuffle=False,
            # collate_fn=collate_fn,
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
            shuffle=False,
            # collate_fn=collate_fn,
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
