import torch
from topobenchmarkx.models.wrappers.wrapper import DefaultWrapper

class CANWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within
    network."""

    def forward(self, batch):
        """Define logic for forward pass."""

        x_1 = self.backbone(
            x_0=batch.x_0,
            x_1=batch.x_1,
            adjacency_0=batch.adjacency_0.coalesce(),
            down_laplacian_1=batch.down_laplacian_1.coalesce(),
            up_laplacian_1=batch.up_laplacian_1.coalesce(),
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_1"] = x_1
        model_out["x_0"] = torch.sparse.mm(batch.incidence_1, x_1)
        return model_out