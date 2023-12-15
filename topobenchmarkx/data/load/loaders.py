import hydra
import torch
from omegaconf import DictConfig

from topobenchmarkx.data.load.loader import AbstractLoader
from topobenchmarkx.data.utils import load_hypergraph_pickle_dataset, load_split


class HypergraphLoader(AbstractLoader):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def load(
        self,
    ):
        data = load_hypergraph_pickle_dataset(self.cfg.data)
        data = load_split(data, self.cfg.data)

        # We need to add checks that:
        # All nodes belong to some edge, in case some not, create selfedge

        return data
