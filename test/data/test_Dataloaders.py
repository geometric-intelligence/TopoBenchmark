"""Test the collate function."""
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

import torch

from topobenchmarkx.data.dataloaders import to_data_list, DefaultDataModule

from topobenchmarkx.utils.config_resolvers import (
    get_default_transform,
    get_monitor_metric,
    get_monitor_mode,
    infer_in_channels,
)

import rootutils

rootutils.setup_root("./", indicator=".project-root", pythonpath=True)

class TestCollateFunction:
    """Test collate_fn."""

    def setup_method(self):
        """Setup the test.
        
        For this test we load the MUTAG dataset.
        
        Parameters
        ----------
        None
        """
        OmegaConf.register_new_resolver("get_default_transform", get_default_transform)
        OmegaConf.register_new_resolver("get_monitor_metric", get_monitor_metric)
        OmegaConf.register_new_resolver("get_monitor_mode", get_monitor_mode)
        OmegaConf.register_new_resolver("infer_in_channels", infer_in_channels)
        OmegaConf.register_new_resolver(
            "parameter_multiplication", lambda x, y: int(int(x) * int(y))
        )

        initialize(version_base="1.3", config_path="../../configs", job_name="job")
        cfg = compose(config_name="train.yaml")
        
        graph_loader = hydra.utils.instantiate(cfg.dataset, _recursive_=False)
        datasets = graph_loader.load()
        self.batch_size = 2
        datamodule = DefaultDataModule(
            dataset_train=datasets[0],
            dataset_val=datasets[1],
            dataset_test=datasets[2],
            batch_size=self.batch_size
        )
        self.val_dataloader = datamodule.val_dataloader()
        self.val_dataset = datasets[1]
        
    def test_lift_features(self):
        """Test the collate funciton.
        
        To test the collate function we use the DefaultDataModule class to create a dataloader that uses the collate function. We then first check that the batched data has the expected shape. We then convert the batched data back to a list and check that the data in the list is the same as the original data.
        
        Parameters
        ----------
        None
        """
        def check_shape(batch, elems, key):
            """Check that the batched data has the expected shape."""
            if 'x_' in key or 'x'==key:
                rows = 0
                for i in range(len(elems)):
                    rows += elems[i][key].shape[0]
                assert batch[key].shape[0] == rows
                assert batch[key].shape[1] == elems[0][key].shape[1]
            elif 'edge_index' in key:
                cols = 0
                for i in range(len(elems)):
                    cols += elems[i][key].shape[1]
                assert batch[key].shape[0] == 2
                assert batch[key].shape[1] == cols
            elif 'batch_' in key:
                rows = 0
                n = int(key.split('_')[1])
                for i in range(len(elems)):
                    rows += elems[i][f'x_{n}'].shape[0]
                assert batch[key].shape[0] == rows
            elif key in elems[0].keys():
                for i in range(len(batch[key].shape)):
                    i_elems = 0
                    for j in range(len(elems)):
                        i_elems += elems[j][key].shape[i]
                    assert batch[key].shape[i] == i_elems
                    
        def check_separation(matrix, n_elems_0_row, n_elems_0_col):
            """Check that the matrix is separated into two parts diagonally concatenated."""
            assert torch.all(matrix[:n_elems_0_row, n_elems_0_col:] == 0)
            assert torch.all(matrix[n_elems_0_row:, :n_elems_0_col] == 0)
            
        def check_values(matrix, m1, m2):
            """Check that the values in the matrix are the same as the values in the original data."""
            assert torch.allclose(matrix[:m1.shape[0], :m1.shape[1]], m1)
            assert torch.allclose(matrix[m1.shape[0]:, m1.shape[1]:], m2)
            
        
        batch = next(iter(self.val_dataloader))
        elems = []
        for i in range(self.batch_size):
            elems.append(self.val_dataset.data_lst[i])

        # Check shape
        for key in batch.keys():
            check_shape(batch, elems, key)

        # Check that the batched data is separated correctly and the values are correct
        if self.batch_size == 2:
            for key in batch.keys():
                if 'incidence_' in key:
                    i = int(key.split('_')[1])
                    if i==0:
                        n0_row = 1
                    else:
                        n0_row = torch.sum(batch[f'batch_{i-1}']==0)
                    n0_col = torch.sum(batch[f'batch_{i}']==0)
                    check_separation(batch[key].to_dense(), n0_row, n0_col)
                    check_values(batch[key].to_dense(), 
                                 elems[0][key].to_dense(), 
                                 elems[1][key].to_dense())

        # Check that going back to a list of data gives the same data
        batch_list = to_data_list(batch)
        assert len(batch_list) == len(elems)
        for i in range(len(batch_list)):
            for key in elems[i].keys():
                if key in batch_list[i].keys():
                    if batch_list[i][key].is_sparse:
                        assert torch.all(batch_list[i][key].coalesce().indices() == elems[i][key].coalesce().indices())
                        assert torch.allclose(batch_list[i][key].coalesce().values(), elems[i][key].coalesce().values())
                        assert batch_list[i][key].shape, elems[i][key].shape
                    else:
                        assert torch.allclose(batch_list[i][key], elems[i][key])