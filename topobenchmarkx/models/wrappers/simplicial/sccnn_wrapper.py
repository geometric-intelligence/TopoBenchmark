from topobenchmarkx.models.wrappers.wrapper import DefaultWrapper

class SCCNNWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within
    network."""

    def forward(self, batch):
        """Define logic for forward pass."""

        x_all = (batch.x_0, batch.x_1, batch.x_2)
        laplacian_all = (
            batch.hodge_laplacian_0,
            batch.down_laplacian_1,
            batch.up_laplacian_1,
            batch.down_laplacian_2,
            batch.up_laplacian_2,
        )

        incidence_all = (batch.incidence_1, batch.incidence_2)
        x_0, x_1, x_2 = self.backbone(x_all, laplacian_all, incidence_all)

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}

        model_out["x_0"] = x_0
        model_out["x_1"] = x_1
        model_out["x_2"] = x_2

        return model_out