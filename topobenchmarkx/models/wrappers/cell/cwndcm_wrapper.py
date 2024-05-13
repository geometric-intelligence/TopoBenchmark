import torch
from topobenchmarkx.models.wrappers.wrapper import DefaultWrapper

class CWNDCMWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within
    network."""

    def forward(self, batch):
        """Define logic for forward pass."""

        x_1 = self.backbone(
            batch.x_1,
            batch.down_laplacian_1.coalesce(),
            batch.up_laplacian_1.coalesce(),
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}

        model_out["x_1"] = x_1
        model_out["x_0"] = torch.sparse.mm(batch.incidence_1, x_1)
        return model_out