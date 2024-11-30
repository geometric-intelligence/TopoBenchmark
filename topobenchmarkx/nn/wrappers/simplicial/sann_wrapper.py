"""Wrapper for the SANN model."""

from topobenchmarkx.nn.wrappers import AbstractWrapper


class SANNWrapper(AbstractWrapper):
    r"""Wrapper for the SANN."""

    def __call__(self, batch):
        r"""Forward pass for the model.

        This method calls the forward method and adds the residual connection.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the model output.
        """
        model_out = self.forward(batch)
        return model_out

    def forward(self, batch):
        """Forward pass of the SANN.

        Parameters
        ----------
        batch : Dict
            Dictionary containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the model output.
        """
        hop_data_names = [
            k
            for k, v in batch
            if k.startswith("x") and len(k.split("_")[0]) == 2
        ]
        max_simplex_dim = max(
            [int(k.split("_")[0][1]) for k in hop_data_names]
        )
        max_hop_dim = max([int(k.split("_")[1][0]) for k in hop_data_names])

        # Prepare the input data for the backbone
        # by aggregating the data in a dictionary
        # (source_simplex_dim, (target_simplex_dim, torch.Tensor with embeddings))
        x_all = tuple(
            tuple(batch[f"x{i}_{j}"] for j in range(max_hop_dim + 1))
            for i in range(max_simplex_dim + 1)
        )

        x_out = self.backbone(x_all)

        model_out = {
            "labels": batch.y,
            "batch_0": batch.batch_0,
            "batch_1": batch.batch_1,
            "batch_2": batch.batch_2,
        }

        model_out["x_0"] = x_out[0]
        model_out["x_1"] = x_out[1]
        model_out["x_2"] = x_out[2]

        return model_out
