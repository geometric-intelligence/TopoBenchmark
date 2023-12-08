from abc import ABC, abstractmethod

import hydra
from omegaconf import DictConfig

# logger = logging.getLogger(__name__)


class AbstractLoss(ABC):
    """Abstract class that provides an interface to loss logic within netowrk"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @abstractmethod
    def init_loss(
        self,
    ):
        """Initialize loss"""

    @abstractmethod
    def forward(self, model_output):
        """Loss logic based on model_output"""
