from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.mixin import (
    AffinityMixin,
    LogMemoryMixin,
    MultithreadingMixin,
)

from topobenchmark.data.batching.utils import filter_data

from torch_geometric.loader.utils import (
    filter_custom_hetero_store,
    filter_custom_store,
    filter_hetero_data,
    get_input_nodes,
    infer_filter_per_worker,
)
from torch_geometric.sampler import (
    BaseSampler,
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.typing import InputNodes, OptTensor


class CellLoader(
        torch.utils.data.DataLoader,
        AffinityMixin,
        MultithreadingMixin,
        LogMemoryMixin,
):
    r"""A data loader that performs mini-batch sampling from cell information,
    using a generic :class:`~torch_geometric.sampler.BaseSampler`
    implementation that defines a
    :meth:`~torch_geometric.sampler.BaseSampler.sample_from_nodes` function and
    is supported on the provided input :obj:`data` object.

    Args:
        data (Any): A :class:`~torch_geometric.data.Data`,
            :class:`~torch_geometric.data.HeteroData`, or
            (:class:`~torch_geometric.data.FeatureStore`,
            :class:`~torch_geometric.data.GraphStore`) data object.
        cell_sampler (torch_geometric.sampler.BaseSampler): The sampler
            implementation to be used with this loader.
            Needs to implement
            :meth:`~torch_geometric.sampler.BaseSampler.sample_from_cells`.
            The sampler implementation must be compatible with the input
            :obj:`data` object.
        input_cells (torch.Tensor or str or Tuple[str, torch.Tensor]): The
            indices of seed cells to start sampling from.
            Needs to be either given as a :obj:`torch.LongTensor` or
            :obj:`torch.BoolTensor`.
            If set to :obj:`None`, all cells will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the cell type and cell indices. (default: :obj:`None`)
        input_time (torch.Tensor, optional): Optional values to override the
            timestamp for the input cells given in :obj:`input_cells`. If not
            set, will use the timestamps in :obj:`time_attr` as default (if
            present). The :obj:`time_attr` needs to be set for this to work.
            (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        transform_sampler_output (callable, optional): A function/transform
            that takes in a :class:`torch_geometric.sampler.SamplerOutput` and
            returns a transformed version. (default: :obj:`None`)
        filter_per_worker (bool, optional): If set to :obj:`True`, will filter
            the returned data in each worker's subprocess.
            If set to :obj:`False`, will filter the returned data in the main
            process.
            If set to :obj:`None`, will automatically infer the decision based
            on whether data partially lives on the GPU
            (:obj:`filter_per_worker=True`) or entirely on the CPU
            (:obj:`filter_per_worker=False`).
            There exists different trade-offs for setting this option.
            Specifically, setting this option to :obj:`True` for in-memory
            datasets will move all features to shared memory, which may result
            in too many open file handles. (default: :obj:`None`)
        custom_cls (HeteroData, optional): A custom
            :class:`~torch_geometric.data.HeteroData` class to return for
            mini-batches in case of remote backends. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        cell_sampler: BaseSampler,
        input_cells: InputNodes = None,
        input_time: OptTensor = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: Optional[bool] = None,
        custom_cls: Optional[HeteroData] = None,
        input_id: OptTensor = None,
        **kwargs,
    ):
        if filter_per_worker is None:
            filter_per_worker = infer_filter_per_worker(data)

        self.data = data
        self.cell_sampler = cell_sampler
        self.input_cells = input_cells
        self.input_time = input_time
        self.transform = transform
        self.transform_sampler_output = transform_sampler_output
        self.filter_per_worker = filter_per_worker
        self.custom_cls = custom_cls
        self.input_id = input_id

        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        # Get cell type (or `None` for homogeneous graphs):
        input_type, input_cells, input_id = get_input_nodes(
            data, input_cells, input_id)

        self.input_data = NodeSamplerInput(
            input_id=input_id,
            node=input_cells,
            time=input_time,
            input_type=input_type,
        )

        iterator = range(input_cells.size(0))
        super().__init__(iterator, collate_fn=self.collate_fn, **kwargs)

    def __call__(
        self,
        index: Union[Tensor, List[int]],
    ) -> Union[Data, HeteroData]:
        r"""Samples a subgraph from a batch of input cells."""
        out = self.collate_fn(index)
        if not self.filter_per_worker:
            out = self.filter_fn(out)
        return out

    def collate_fn(self, index: Union[Tensor, List[int]]) -> Any:
        r"""Samples a subgraph from a batch of input cells."""
        input_data: NodeSamplerInput = self.input_data[index]

        out = self.cell_sampler.sample_from_nodes(input_data)

        if self.filter_per_worker:  # Execute `filter_fn` in the worker process
            out = self.filter_fn(out)

        return out

    def filter_fn(
        self,
        out: Union[SamplerOutput, HeteroSamplerOutput],
    ) -> Union[Data, HeteroData]:
        r"""Joins the sampled cells with their corresponding features,
        returning the resulting :class:`~torch_geometric.data.Data` 
        object to be used downstream.
        """
        if self.transform_sampler_output:
            out = self.transform_sampler_output(out)

        if isinstance(out, SamplerOutput) and isinstance(self.data, Data):
            data = filter_data(  #
                self.data, out.node, self.rank)
        else:
            raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                            f"type: '{type(data)}'")

        return data if self.transform is None else self.transform(data)

    def _get_iterator(self) -> Iterator:
        if self.filter_per_worker:
            return super()._get_iterator()

        # if not self.is_cuda_available and not self.cpu_affinity_enabled:
        # TODO: Add manual page for best CPU practices
        # link = ...
        # Warning('Dataloader CPU affinity opt is not enabled, consider '
        #          'switching it on with enable_cpu_affinity() or see CPU '
        #          f'best practices for PyG [{link}])')

        # Execute `filter_fn` in the main process:
        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'