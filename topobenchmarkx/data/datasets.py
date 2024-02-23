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


class PreprocessedDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, data_list, pre_transform=None):
        if isinstance(data_list, torch_geometric.data.Dataset):
            data_list = [data_list.get(idx) for idx in range(len(data_list))]
        elif isinstance(data_list, torch_geometric.data.Data):
            data_list = [data_list]
        self.data_list = data_list
        # The self.pocess is called in the super().__init__
        super().__init__(root, None, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return self.root

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):
        if self.pre_transform is not None:
            self.data_list = [self.pre_transform(d) for d in self.data_list]

        self.data, self.slices = self.collate(self.data_list)
        self._data_list = None  # Reset cache.

        assert isinstance(self._data, torch_geometric.data.Data)
        self.save(self.data_list, self.processed_paths[0])
