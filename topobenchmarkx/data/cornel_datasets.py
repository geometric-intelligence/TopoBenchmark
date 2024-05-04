# import os.path as osp
# from typing import Callable, List, Optional

# from torch_geometric.data import Data, InMemoryDataset
# from torch_geometric.io import fs, read_tu_data


# class CornelDataset(InMemoryDataset):
#     r"""
#     """

#     URLS = {'contact-high-school': 'https://drive.google.com/open?id=1VA2P62awVYgluOIh1W4NZQQgkQCBk-Eu'}
#     file_format = "tar.gz"

#     def __init__(
#         self,
#         root: str,
#         name: str,
#         transform: Optional[Callable] = None,
#         pre_transform: Optional[Callable] = None,
#         pre_filter: Optional[Callable] = None,
#         force_reload: bool = False,
#         use_node_attr: bool = False,
#         use_edge_attr: bool = False,
#         cleaned: bool = False,
#     ) -> None:
#         self.name = name
#         self.cleaned = cleaned
#         super().__init__(root, transform, pre_transform, pre_filter,
#                          force_reload=force_reload)

#         out = fs.torch_load(self.processed_paths[0])
#         if not isinstance(out, tuple) or len(out) < 3:
#             raise RuntimeError(
#                 "The 'data' object was created by an older version of PyG. "
#                 "If this error occurred while loading an already existing "
#                 "dataset, remove the 'processed/' directory in the dataset's "
#                 "root folder and try again.")
#         assert len(out) == 3 or len(out) == 4

#         if len(out) == 3:  # Backward compatibility.
#             data, self.slices, self.sizes = out
#             data_cls = Data
#         else:
#             data, self.slices, self.sizes, data_cls = out

#         if not isinstance(data, dict):  # Backward compatibility.
#             self.data = data
#         else:
#             self.data = data_cls.from_dict(data)

#         assert isinstance(self._data, Data)
#         if self._data.x is not None and not use_node_attr:
#             num_node_attributes = self.num_node_attributes
#             self._data.x = self._data.x[:, num_node_attributes:]
#         if self._data.edge_attr is not None and not use_edge_attr:
#             num_edge_attrs = self.num_edge_attributes
#             self._data.edge_attr = self._data.edge_attr[:, num_edge_attrs:]

#     @property
#     def raw_dir(self) -> str:
#         name = f'raw{"_cleaned" if self.cleaned else ""}'
#         return osp.join(self.root, self.name, name)

#     @property
#     def processed_dir(self) -> str:
#         name = f'processed{"_cleaned" if self.cleaned else ""}'
#         return osp.join(self.root, self.name, name)

#     @property
#     def num_node_labels(self) -> int:
#         return self.sizes['num_node_labels']

#     @property
#     def num_node_attributes(self) -> int:
#         return self.sizes['num_node_attributes']

#     @property
#     def num_edge_labels(self) -> int:
#         return self.sizes['num_edge_labels']

#     @property
#     def num_edge_attributes(self) -> int:
#         return self.sizes['num_edge_attributes']

#     @property
#     def raw_file_names(self) -> List[str]:
#         names = ['A', 'graph_indicator']
#         return [f'{self.name}_{name}.txt' for name in names]

#     @property
#     def processed_file_names(self) -> str:
#         return 'data.pt'

#     def download(self) -> None:
#         from topobenchmarkx.io.load.download_utils import download_google_drive_datasets
#         self.url = self.URLS[self.name] 
        
#         download_google_drive_datasets(
#             self.url, 
#             path_to_save=self.raw_dir, 
#             dataset_name=self.name,
#             file_format=self.file_format
#         )

#         fs.cp(f'{self.raw_dir}/{self.name}.{self.file_format}', self.raw_dir, extract=True)

#         # url = self.cleaned_url if self.cleaned else self.url
#         # fs.cp(f'{url}/{self.name}.zip', self.raw_dir, extract=True)


#         # for filename in fs.ls(osp.join(self.raw_dir, self.name)):
#         #     fs.mv(filename, osp.join(self.raw_dir, osp.basename(filename)))
#         # fs.rm(osp.join(self.raw_dir, self.name))

#     def process(self) -> None:
#         self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)

#         if self.pre_filter is not None or self.pre_transform is not None:
#             data_list = [self.get(idx) for idx in range(len(self))]

#             if self.pre_filter is not None:
#                 data_list = [d for d in data_list if self.pre_filter(d)]

#             if self.pre_transform is not None:
#                 data_list = [self.pre_transform(d) for d in data_list]

#             self.data, self.slices = self.collate(data_list)
#             self._data_list = None  # Reset cache.

#         assert isinstance(self._data, Data)
#         fs.torch_save(
#             (self._data.to_dict(), self.slices, sizes, self._data.__class__),
#             self.processed_paths[0],
#         )

#     def __repr__(self) -> str:
#         return f'{self.name}({len(self)})'


import sys
# Add manually root '/home/lev/projects/TopoBenchmarkX'
root_path = '/home/lev/projects/TopoBenchmarkX'
if root_path not in sys.path:
    sys.path.append(root_path)

import os.path as osp
from typing import Callable, List, Optional

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs, read_tu_data

from topobenchmarkx.io.load.download_utils import download_file_from_drive

class CornelDataset(InMemoryDataset):
    r"""
    """

    URLS = {
        'contact-high-school': 'https://drive.google.com/open?id=1VA2P62awVYgluOIh1W4NZQQgkQCBk-Eu',
        'US-county-demos-2012': 'https://drive.google.com/file/d/1FNF_LbByhYNICPNdT6tMaJI9FxuSvvLK/view?usp=sharing',
        'US-county-demos-2016': 'https://drive.google.com/file/d/1FNF_LbByhYNICPNdT6tMaJI9FxuSvvLK/view?usp=sharing'
    }

    FILE_FORMAT = {
        'contact-high-school': 'tar.gz',
        'US-county-demos': 'zip'
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        use_node_attr: bool = False,
        use_edge_attr: bool = False,
        cleaned: bool = False,
    ) -> None:
        self.name = name.replace('_', '-')
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        out = fs.torch_load(self.processed_paths[0])
        if not isinstance(out, tuple) or len(out) < 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:  # Backward compatibility.
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        assert isinstance(self._data, Data)
        if self._data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self._data.x = self._data.x[:, num_node_attributes:]
        if self._data.edge_attr is not None and not use_edge_attr:
            num_edge_attrs = self.num_edge_attributes
            self._data.edge_attr = self._data.edge_attr[:, num_edge_attrs:]

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        return self.sizes['num_node_labels']

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_labels(self) -> int:
        return self.sizes['num_edge_labels']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        # Download data
        self.url = self.URLS[self.name] 
        self.file_format = self.FILE_FORMAT[self.name]
        
        download_file_from_drive(
            file_link=self.url, 
            path_to_save=self.raw_dir, 
            dataset_name=self.name,
            file_format=self.file_format
        )

        fs.cp(f'{self.raw_dir}/{self.name}.{self.file_format}', self.raw_dir, extract=True)

        # Move into raw/
        for filename in fs.ls(osp.join(self.raw_dir, self.name)):
            fs.mv(filename, osp.join(self.raw_dir, osp.basename(filename)))
        fs.rm(osp.join(self.raw_dir, self.name))

        # Delete also f'{self.raw_dir}/{self.name}.{self.file_format}'
        fs.rm(f'{self.raw_dir}/{self.name}.{self.file_format}')

    def process(self) -> None:
        data = load_us_county_demos(self.raw_dir, self.name)

        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'