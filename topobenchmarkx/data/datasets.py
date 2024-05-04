import torch_geometric


class CustomDataset(torch_geometric.data.Dataset):
    def __init__(self, data_lst):
        super().__init__()
        self.data_lst = data_lst

    def get(self, idx):
        data = self.data_lst[idx]
        keys = list(data.keys())
        return ([data[key] for key in keys], keys)

    def len(self):
        return len(self.data_lst)


class TorchGeometricDataset(torch_geometric.data.Dataset):
    def __init__(self, data_lst):
        super().__init__()
        self.data_lst = data_lst

    def get(self, idx):
        data = self.data_lst[idx]

        return data

    def len(self):
        return len(self.data_lst)


