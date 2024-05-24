from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class DefaultWrapper(ABC, torch.nn.Module):
    """Abstract class that provides an interface to handle the network
    output."""

    def __init__(self, backbone, **kwargs):
        super().__init__()
        self.backbone = backbone
        out_channels = kwargs["out_channels"]
        self.dimensions = range(kwargs["num_cell_dimensions"])

        for i in self.dimensions:
            setattr(
                self,
                f"ln_{i}",
                nn.LayerNorm(out_channels),
            )

    def __call__(self, batch):
        """Define logic for forward pass."""
        model_out = self.forward(batch)
        model_out = self.residual_connection(model_out=model_out, batch=batch)
        return model_out

    def residual_connection(self, model_out, batch):
        for i in self.dimensions:
            if (
                (f"x_{i}" in batch)
                and hasattr(self, f"ln_{i}")
                and (f"x_{i}" in model_out)
            ):
                residual = model_out[f"x_{i}"] + batch[f"x_{i}"]
                model_out[f"x_{i}"] = getattr(self, f"ln_{i}")(residual)
        return model_out

    @abstractmethod
    def forward(self, batch):
        """Define handling output here."""


class GNNWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within
    network."""

    # def __init__(self, backbone, **kwargs):
    #     super().__init__(backbone)

    def forward(self, batch):
        """Define logic for forward pass."""
        x_0 = self.backbone(batch.x_0, batch.edge_index)

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0

        return model_out


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


class SANWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within
    network."""

    def forward(self, batch):
        """Define logic for forward pass."""
        x_1 = self.backbone(
            batch.x_1, batch.up_laplacian_1, batch.down_laplacian_1
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = torch.sparse.mm(batch.incidence_1, x_1)
        model_out["x_1"] = x_1
        return model_out


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


class CCXNWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within
    network."""

    def forward(self, batch):
        """Define logic for forward pass."""

        x_0, x_1, x_2 = self.backbone(
            x_0=batch.x_0,
            x_1=batch.x_1,
            adjacency_0=batch.adjacency_0,
            incidence_2_t=batch.incidence_2.T,
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0
        model_out["x_1"] = x_1
        model_out["x_2"] = x_2
        return model_out
