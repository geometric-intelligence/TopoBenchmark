import torch
from topobenchmarkx.models.wrappers.wrapper import DefaultWrapper

class SCNWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within
    network."""

    def forward(self, batch):
        """Define logic for forward pass."""

        laplacian_0 = self.normalize_matrix(batch.hodge_laplacian_0)
        laplacian_1 = self.normalize_matrix(batch.hodge_laplacian_1)
        laplacian_2 = self.normalize_matrix(batch.hodge_laplacian_2)
        x_0, x_1, x_2 = self.backbone(
            batch.x_0,
            batch.x_1,
            batch.x_2,
            laplacian_0,
            laplacian_1,
            laplacian_2,
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_2"] = x_2
        model_out["x_1"] = x_1
        model_out["x_0"] = x_0

        return model_out

    def normalize_matrix(self, matrix):
        matrix_ = matrix.to_dense()
        n, _ = matrix_.shape
        abs_matrix = abs(matrix_)
        diag_sum = abs_matrix.sum(axis=1)

        # Handle division by zero
        idxs = torch.where(diag_sum != 0)
        diag_sum[idxs] = 1.0 / torch.sqrt(diag_sum[idxs])

        diag_indices = torch.stack([torch.arange(n), torch.arange(n)])
        diag_matrix = torch.sparse_coo_tensor(
            diag_indices, diag_sum, matrix_.shape, device=matrix.device
        ).coalesce()
        normalized_matrix = diag_matrix @ (matrix @ diag_matrix)
        return normalized_matrix