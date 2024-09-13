"""Main entry point for training and testing models."""

from typing import Any

import hydra
import numpy as np
import rootutils
import torch
import torch_geometric
from omegaconf import DictConfig, OmegaConf, open_dict

from topobenchmarkx.data.preprocessor import PreProcessor
from topobenchmarkx.dataloader import TBXDataloader
from topobenchmarkx.utils import (
    RankedLogger,
)
from topobenchmarkx.utils.config_resolvers import (
    get_default_metrics,
    get_default_transform,
    get_monitor_metric,
    get_monitor_mode,
    get_required_lifting,
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


OmegaConf.register_new_resolver(
    "get_default_metrics", get_default_metrics, replace=True
)
OmegaConf.register_new_resolver(
    "get_default_transform", get_default_transform, replace=True
)
OmegaConf.register_new_resolver(
    "get_required_lifting", get_required_lifting, replace=True
)
OmegaConf.register_new_resolver(
    "get_monitor_metric", get_monitor_metric, replace=True
)
OmegaConf.register_new_resolver(
    "get_monitor_mode", get_monitor_mode, replace=True
)
OmegaConf.register_new_resolver(
    "infer_in_channels", infer_in_channels, replace=True
)
OmegaConf.register_new_resolver(
    "infere_num_cell_dimensions", infere_num_cell_dimensions, replace=True
)
OmegaConf.register_new_resolver(
    "parameter_multiplication", lambda x, y: int(int(x) * int(y)), replace=True
)


def initialize_hydra() -> DictConfig:
    """Initialize Hydra when main is not an option (e.g. tests).

    Returns
    -------
    DictConfig
        A DictConfig object containing the config tree.
    """
    hydra.initialize(
        version_base="1.3", config_path="../configs", job_name="run"
    )
    cfg = hydra.compose(config_name="run.yaml")
    return cfg


torch.set_num_threads(1)
log = RankedLogger(__name__, rank_zero_only=True)


def run(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the model.

    Can additionally evaluate on a testset, using best weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls
    the behavior during failure. Useful for multiruns, saving info about the
    crash, etc.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        A tuple with metrics and dict with all instantiated objects.
    """
    cfg.transforms.graph2simplicial_lifting.complex_dim = 3
    cfg.transforms.graph2simplicial_lifting.signed = False

    # Load curvature lifting
    node_degrees_transform = OmegaConf.load(
        "./configs/transforms/data_manipulations/node_degrees.yaml"
    )
    node_degrees_transform["selected_fields"].append("incidence")

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.transforms.node_degrees_transform = node_degrees_transform

    simplicial_curvature_transform = OmegaConf.load(
        "./configs/transforms/data_manipulations/simplicial_curvature.yaml"
    )
    with open_dict(cfg):
        cfg.transforms.simplicial_curvature_transform = (
            simplicial_curvature_transform
        )

    keel_selected_fiels = OmegaConf.load(
        "./configs/transforms/data_manipulations/keep_selected_fields.yaml"
    )
    with open_dict(cfg):
        cfg.transforms.keel_selected_fiels = keel_selected_fiels

    dataset_loader = hydra.utils.instantiate(cfg.dataset.loader)
    dataset, dataset_dir = dataset_loader.load()

    # Preprocess dataset and load the splits
    # log.info("Instantiating preprocessor...")
    transform_config = cfg.get("transforms", None)
    preprocessor = PreProcessor(dataset, dataset_dir, transform_config)
    dataset_train, dataset_val, dataset_test = (
        preprocessor.load_dataset_splits(cfg.dataset.split_params)
    )
    # Prepare datamodule
    # log.info("Instantiating datamodule...")
    if cfg.dataset.parameters.task_level in ["node", "graph"]:
        datamodule = TBXDataloader(
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            dataset_test=dataset_test,
            **cfg.dataset.get("dataloader_params", {}),
        )
    else:
        raise ValueError("Invalid task_level")

    if len(datamodule.train_dataloader()) > 1:
        data = torch_geometric.data.Data()

        d = {
            "0_cell_curvature": [],
            "1_cell_curvature": [],
            "2_cell_curvature": [],
        }
        for batch in datamodule.train_dataloader():
            d["0_cell_curvature"].append(batch["0_cell_curvature"])
            d["1_cell_curvature"].append(batch["1_cell_curvature"])
            d["2_cell_curvature"].append(batch["2_cell_curvature"])

        for batch in datamodule.test_dataloader():
            d["0_cell_curvature"].append(batch["0_cell_curvature"])
            d["1_cell_curvature"].append(batch["1_cell_curvature"])
            d["2_cell_curvature"].append(batch["2_cell_curvature"])

        for batch in datamodule.val_dataloader():
            d["0_cell_curvature"].append(batch["0_cell_curvature"])
            d["1_cell_curvature"].append(batch["1_cell_curvature"])
            d["2_cell_curvature"].append(batch["2_cell_curvature"])

        data["0_cell_curvature"] = torch.cat(d["0_cell_curvature"], dim=0)
        data["1_cell_curvature"] = torch.cat(d["1_cell_curvature"], dim=0)
        data["2_cell_curvature"] = torch.cat(d["2_cell_curvature"], dim=0)

    else:
        data = next(iter(datamodule.train_dataloader()))

    import matplotlib.pyplot as plt

    def log10(a):
        b = np.zeros(a.shape)
        positive_idx = a > 0
        zeros_idx = a == 0
        b[positive_idx] = np.log10(a[positive_idx])
        b[~positive_idx] = np.log10(0 + 1e-6) - np.log10(-a[~positive_idx])
        b[zeros_idx] = np.log10(a[zeros_idx] + 1e-6)
        return b

    # Create subplots with one row and three columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the histogram 0_cell_curvature and remove outliers
    zero_cell_curvature = data["0_cell_curvature"].numpy().flatten()

    axes[0].hist(zero_cell_curvature, bins=100)
    axes[0].set_title("Histogram 0_cell_curvature")

    # Plot the histogram 1_cell_curvature and remove outliers
    one_cell_curvature = data["1_cell_curvature"].numpy().flatten()
    axes[1].hist(one_cell_curvature, bins=100)
    axes[1].set_title("Histogram 1_cell_curvature")

    # Plot the histogram 2_cell_curvature and remove outliers
    two_cell_curvature = data["2_cell_curvature"].numpy().flatten()
    axes[2].hist(two_cell_curvature, bins=100)
    axes[2].set_title("Histogram 2_cell_curvature")

    # Add titpe to every subplot
    for i, ax in enumerate(axes):
        ax.set_xlabel("Curvature")
        ax.set_yscale("log")
        if i == 0:
            ax.set_ylabel("Frequency")

    # Add figure title
    fig.suptitle(f"{cfg.dataset.loader.parameters.data_name}")

    # Save the figure
    plt.savefig(
        f"plots/{cfg.dataset.loader.parameters.data_name}_curvature_histogram.png"
    )
    return


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="run.yaml"
)
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    float | None
        Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)

    # train the model
    run(cfg)

    return


if __name__ == "__main__":
    main()
