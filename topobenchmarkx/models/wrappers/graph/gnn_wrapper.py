from topobenchmarkx.models.wrappers.wrapper import DefaultWrapper

class GNNWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within
    network."""

    def forward(self, batch):
        """Define logic for forward pass."""
        x_0 = self.backbone(batch.x_0, batch.edge_index)

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0

        return model_out