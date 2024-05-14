import torch
import torch_geometric
import topomodelx
from topobenchmarkx.models.readouts.readout import AbstractReadOut

class PropagateSignalDown(AbstractReadOut):
    def __init__(self, **kwargs):
        super().__init__()

        self.name = kwargs["readout_name"]
        self.dimensions = range(kwargs["num_cell_dimensions"] - 1, 0, -1)
        hidden_dim = kwargs["hidden_dim"]

        for i in self.dimensions:
            setattr(
                self,
                f"agg_conv_{i}",
                topomodelx.base.conv.Conv(
                    hidden_dim, hidden_dim, aggr_norm=False
                ),
            )

            setattr(self, f"ln_{i}", torch.nn.LayerNorm(hidden_dim))

            setattr(
                self,
                f"projector_{i}",
                torch.nn.Linear(2 * hidden_dim, hidden_dim),
            )

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        for i in self.dimensions:
            x_i = getattr(self, f"agg_conv_{i}")(
                model_out[f"x_{i}"], batch[f"incidence_{i}"]
            )
            x_i = getattr(self, f"ln_{i}")(x_i)
            model_out[f"x_{i-1}"] = getattr(self, f"projector_{i}")(
                torch.cat([x_i, model_out[f"x_{i-1}"]], dim=1)
            )

        return model_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_cell_dimensions={len(self.dimensions)}, hidden_dim={self.hidden_dim}, readout_name={self.name}"
