import torch_geometric


class CustomDataset(torch_geometric.data.Dataset):
    r"""Custom dataset to return all the values added to the dataset object.

    Parameters
    ----------
    data_lst: list
        List of torch_geometric.data.Data objects .
    """
    def __init__(self, data_lst):
        super().__init__()
        self.data_lst = data_lst

    def get(self, idx):
        r"""Get data object from data list.

        Parameters
        ----------
        idx: int
            Index of the data object to get.

        Returns
        -------
        tuple
            tuple containing a list of all the values for the data and the keys corresponding to the values.
        """
        data = self.data_lst[idx]
        keys = list(data.keys())
        return ([data[key] for key in keys], keys)

    def len(self):
        r"""Return length of the dataset.
        
        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data_lst)


class TorchGeometricDataset(torch_geometric.data.Dataset):
    r"""Dataset to work with a list of data objects.

    Parameters
    ----------
    data_lst: list
        List of torch_geometric.data.Data objects .
    """
    def __init__(self, data_lst):
        super().__init__()
        self.data_lst = data_lst

    def get(self, idx):
        r"""Get data object from data list.

        Parameters
        ----------
        idx: int
            Index of the data object to get.

        Returns
        -------
        torch_geometric.data.Data
            Data object of corresponding index.
        """
        data = self.data_lst[idx]
        return data

    def len(self):
        r"""Return length of the dataset.
        
        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data_lst)
