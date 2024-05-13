from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class DefaultWrapper(ABC, torch.nn.Module):
    """Abstract class that provides an interface to handle the network
    output."""

    def __init__(self, backbone, **kwargs):
        super().__init__()
        self.backbone = backbone
        out_channels = kwargs["out_channels"]
        self.dimensions = range(kwargs["num_cell_dimensions"])

        for i in self.dimensions:
            setattr(
                self,
                f"ln_{i}",
                nn.LayerNorm(out_channels),
            )

    def __call__(self, batch):
        """Define logic for forward pass."""
        model_out = self.forward(batch)
        model_out = self.residual_connection(model_out=model_out, batch=batch)
        return model_out

    def residual_connection(self, model_out, batch):
        for i in self.dimensions:
            if (
                (f"x_{i}" in batch)
                and hasattr(self, f"ln_{i}")
                and (f"x_{i}" in model_out)
            ):
                residual = model_out[f"x_{i}"] + batch[f"x_{i}"]
                model_out[f"x_{i}"] = getattr(self, f"ln_{i}")(residual)
        return model_out

    @abstractmethod
    def forward(self, batch):
        """Define handling output here."""
        pass
