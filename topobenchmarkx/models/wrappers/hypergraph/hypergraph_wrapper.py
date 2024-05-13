from topobenchmarkx.models.wrappers.wrapper import DefaultWrapper

class HypergraphWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within
    network."""

    def forward(self, batch):
        """Define logic for forward pass."""
        x_0, x_1 = self.backbone(batch.x_0, batch.incidence_hyperedges)
        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0
        model_out["hyperedge"] = x_1

        return model_out