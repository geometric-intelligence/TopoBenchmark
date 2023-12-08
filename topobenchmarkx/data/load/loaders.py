import hydra
from omegaconf import DictConfig

from topobenchmarkx.data.load.loader import AbstractLoader
from topobenchmarkx.data.utils import load_hypergraph_coauthorhip_dataset


class HypergraphLoader(AbstractLoader):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def load(
        self,
    ):
        data = load_hypergraph_coauthorhip_dataset(self.cfg.data)
        return data
