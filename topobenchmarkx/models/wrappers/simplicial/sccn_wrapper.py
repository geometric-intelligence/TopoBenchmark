from topobenchmarkx.models.wrappers.wrapper import DefaultWrapper

class SCCNWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within
    network."""

    def forward(self, batch):
        """Define logic for forward pass."""

        features = {
            f"rank_{r}": batch[f"x_{r}"]
            for r in range(self.backbone.layers[0].max_rank + 1)
        }
        incidences = {
            f"rank_{r}": batch[f"incidence_{r}"]
            for r in range(1, self.backbone.layers[0].max_rank + 1)
        }
        adjacencies = {
            f"rank_{r}": batch[f"hodge_laplacian_{r}"]
            for r in range(self.backbone.layers[0].max_rank + 1)
        }
        output = self.backbone(features, incidences, adjacencies)

        # TODO: First decide which strategy is the best then make code general
        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        if len(output) == 3:
            x_0, x_1, x_2 = (
                output["rank_0"],
                output["rank_1"],
                output["rank_2"],
            )

            model_out["x_2"] = x_2
            model_out["x_1"] = x_1
            model_out["x_0"] = x_0

        elif len(output) == 2:
            x_0, x_1 = output["rank_0"], output["rank_1"]

            model_out["x_1"] = x_1
            model_out["x_0"] = x_0

        else:
            raise ValueError(
                f"Invalid number of output tensors: {len(output)}"
            )

        return model_out