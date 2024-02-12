from abc import ABC, abstractmethod

import torch


class AbstractReadOut(torch.nn.Module):
    """abstract class that provides an interface to define a custom readout"""

    def __init__(self):
        return

    @abstractmethod
    def forward(self, model_out: dict) -> dict:
        """Forward pass of the readout model

        Parameters:
            :model_out: Dictionary with results from the network

        Returns:
            :model_out: Dictionary updated with "logits" field

        """
