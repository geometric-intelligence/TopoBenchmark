"""Test the GraphLoader class."""

from unittest.mock import MagicMock, patch
import torch
import pytest
from omegaconf import DictConfig, OmegaConf

from topobenchmarkx.data.loaders import GraphLoader

import os
import hydra

class TestLoaders:
    """Test the GraphLoader class."""
    
    def setup_method(self):
        """Setup the test."""
        base_dir = os.path.abspath(__file__)
        # Go up four levels
        for _ in range(4):
            base_dir = os.path.dirname(base_dir)
        
        config_dir = os.path.join(base_dir, "configs/dataset/graph")

        self.config_files = [f for f in os.listdir(config_dir) if f.endswith(".yaml")]


        # Exclude files 
        exclude_datasets = ["manual_dataset.yaml"]

        self.config_files = [f for f in self.config_files if f not in exclude_datasets]
        self.relative_config_dir =  "../../../configs"

    def test_init(self):
        """Test the initialization of the GraphLoader class."""

        for f in self.config_files:
            print(f)
            with hydra.initialize(version_base="1.3",
                config_path=self.relative_config_dir,
                job_name="run"
            ):
                parameters = hydra.compose(config_name="run.yaml",
                    overrides=[f"dataset=graph/{f}"], 
                    return_hydra_config=True
                )

            loader = GraphLoader(parameters.dataset.loader.parameters)
            loader.load()