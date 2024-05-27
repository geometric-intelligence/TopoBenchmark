from typing import Any

import torch_geometric
from torch_geometric.utils import is_sparse


class DataloadDataset(torch_geometric.data.Dataset):
    r"""Custom dataset to return all the values added to the dataset object.

    Args:
        data_lst (list[torch_geometric.data.Data]): List of torch_geometric.data.Data objects.
    """

    def __init__(self, data_lst):
        super().__init__()
        self.data_lst = data_lst

    def __repr__(self):
        return f"{self.__class__.__name__}(data_lst={self.data_lst})"
    
    def get(self, idx):
        r"""Get data object from data list.

        Args:
            idx (int): Index of the data object to get.

        Returns:
            tuple: tuple containing a list of all the values for the data and the corresponding keys.
        """
        data = self.data_lst[idx]
        keys = list(data.keys())
        return ([data[key] for key in keys], keys)

    def len(self):
        r"""Return the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data_lst)
    

class DomainData(torch_geometric.data.Data):
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
