import os
import random
from typing import Any

import hydra
import lightning as L
import numpy as np
import pandas as pd
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

from topobenchmarkx.dataloader import TBXDataloader
from topobenchmarkx.utils import (
    RankedLogger,
    extras,
)
from topobenchmarkx.utils.config_resolvers import (
    get_default_transform,
    get_monitor_metric,
    get_monitor_mode,
    infer_in_channels,
    infere_num_cell_dimensions,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #


OmegaConf.register_new_resolver("get_default_transform", get_default_transform)
OmegaConf.register_new_resolver("get_monitor_metric", get_monitor_metric)
OmegaConf.register_new_resolver("get_monitor_mode", get_monitor_mode)
OmegaConf.register_new_resolver("infer_in_channels", infer_in_channels)
OmegaConf.register_new_resolver(
    "infere_num_cell_dimensions", infere_num_cell_dimensions
)
OmegaConf.register_new_resolver(
    "parameter_multiplication", lambda x, y: int(int(x) * int(y))
)

torch.set_num_threads(1)
log = RankedLogger(__name__, rank_zero_only=True)


def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best
    weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls
    the behavior during failure. Useful for multiruns, saving info about the
    crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    # if cfg.get("seed"):
    L.seed_everything(cfg.seed, workers=True)
    # Seed for torch
    torch.manual_seed(cfg.seed)
    # Seed for numpy
    np.random.seed(cfg.seed)
    # Seed for python random
    random.seed(cfg.seed)

    if cfg.model.model_domain == "cell":
        cfg.dataset.transforms.graph2cell_lifting.max_cell_length = 1000

    # Instantiate and load dataset
    dataset = hydra.utils.instantiate(cfg.dataset, _recursive_=False)
    dataset = dataset.load()

    one_graph_flag = True
    if cfg.dataset.parameters.batch_size != 1:
        one_graph_flag = False

    log.info(f"Instantiating datamodule <{cfg.dataset._target_}>")

    if cfg.dataset.parameters.task_level == "node":
        datamodule = TBXDataloader(dataset_train=dataset)

    elif cfg.dataset.parameters.task_level == "graph":
        datamodule = TBXDataloader(
            dataset_train=dataset[0],
            dataset_val=dataset[1],
            dataset_test=dataset[2],
            batch_size=cfg.dataset.parameters.batch_size,
        )

    else:
        raise ValueError("Invalid task_level")

    if one_graph_flag:
        dataloaders = [datamodule.train_dataloader()]
    else:
        dataloaders = [
            datamodule.train_dataloader(),
            datamodule.val_dataloader(),
            datamodule.test_dataloader(),
        ]

    dict_collector = {
        "num_hyperedges": 0,
        "zero_cell": 0,
        "one_cell": 0,
        "two_cell": 0,
        "three_cell": 0,
    }

    cell_dict = {
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 0,
        "greater_than_10": 0,
    }

    for loader in dataloaders:
        for batch in loader:
            if cfg.model.model_domain == "hypergraph":
                dict_collector["zero_cell"] += batch.x.shape[0]
                dict_collector["num_hyperedges"] += batch.x_hyperedges.shape[0]

            elif cfg.model.model_domain == "simplicial":
                dict_collector["zero_cell"] += batch.x_0.shape[0]
                dict_collector["one_cell"] += batch.x_1.shape[0]
                dict_collector["two_cell"] += batch.x_2.shape[0]
                dict_collector["three_cell"] += batch.x_3.shape[0]

            elif cfg.model.model_domain == "cell":
                dict_collector["zero_cell"] += batch.x_0.shape[0]
                dict_collector["one_cell"] += batch.x_1.shape[0]
                dict_collector["two_cell"] += batch.x_2.shape[0]
                cell_sizes, cell_counts = torch.unique(
                    batch.incidence_2.to_dense().sum(0), return_counts=True
                )
                cell_sizes = cell_sizes.long()
                for i in range(len(cell_sizes)):
                    if cell_sizes[i].item() > 10:
                        cell_dict["greater_than_10"] += cell_counts[i].item()
                    else:
                        cell_dict[str(cell_sizes[i].item())] += cell_counts[
                            i
                        ].item()

    # Get current working dir
    filename = f"{cfg.paths['root_dir']}/tables/dataset_statistics.csv"

    dict_collector["dataset"] = cfg.dataset.parameters.data_name
    dict_collector["domain"] = cfg.model.model_domain

    df = pd.DataFrame.from_dict(dict_collector, orient="index")
    if not os.path.exists(filename):
        # Save to csv file such as methods .... is a header
        df.T.to_csv(filename, header=True)
    else:
        # read csv file with deader
        df_saved = pd.read_csv(filename, index_col=0)
        # add new row
        df_saved = df_saved._append(dict_collector, ignore_index=True)
        # write to csv file
        df_saved.to_csv(filename)

    if cfg.model.model_domain == "cell":
        filename = f"{cfg.paths['root_dir']}/tables/cell_statistics.csv"

        cell_dict["dataset"] = cfg.dataset.parameters.data_name
        cell_dict["domain"] = cfg.model.model_domain

        df = pd.DataFrame.from_dict(cell_dict, orient="index")
        if not os.path.exists(filename):
            # Save to csv file such as methods .... is a header
            df.T.to_csv(filename, header=True)
        else:
            # read csv file with deader
            df_saved = pd.read_csv(filename, index_col=0)
            # add new row
            df_saved = df_saved._append(df.T, ignore_index=True)
            # write to csv file
            df_saved.to_csv(filename)


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="run.yaml"
)
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
