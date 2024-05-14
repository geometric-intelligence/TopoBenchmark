
import torch_geometric
from topobenchmarkx.models.readouts.readout import AbstractReadOut


class NoReadOut(AbstractReadOut):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, model_out: dict, batch: torch_geometric.data.Data) -> dict:
        return model_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_cell_dimensions={len(self.dimensions)}, hidden_dim={self.hidden_dim}, readout_name={self.name}"
