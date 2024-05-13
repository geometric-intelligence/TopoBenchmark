from topobenchmarkx.models.wrappers.wrapper import DefaultWrapper

class CWNWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within
    network."""

    def forward(self, batch):
        """Define logic for forward pass."""

        x_0, x_1, x_2 = self.backbone(
            x_0=batch.x_0,
            x_1=batch.x_1,
            x_2=batch.x_2,
            incidence_1_t=batch.incidence_1.T,
            adjacency_0=batch.adjacency_1,
            incidence_2=batch.incidence_2,
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0
        model_out["x_1"] = x_1
        model_out["x_2"] = x_2
        return model_out