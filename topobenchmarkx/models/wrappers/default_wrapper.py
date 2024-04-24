from abc import ABC, abstractmethod

import topomodelx
import torch
from torch_geometric.nn.norm import GraphNorm


class DefaultWrapper(ABC, torch.nn.Module):
    """Abstract class that provides an interface to loss logic within netowrk"""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    @abstractmethod
    def __call__(self, batch):
        """Define logic for forward pass"""
        pass


class GNNWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within network"""

    def __init__(self, backbone):
        super().__init__(backbone)

    def __call__(self, batch):
        """Define logic for forward pass"""
        model_out = {"labels": batch.y}
        x_0 = self.backbone(batch.x_0, batch.edge_index)
        model_out["x_0"] = x_0
        model_out["batch"] = batch.batch
        return model_out


class HypergraphWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within network"""

    def __init__(self, backbone):
        super().__init__(backbone)

    def __call__(self, batch):
        """Define logic for forward pass"""
        model_out = {"labels": batch.y, "batch": batch.batch}
        x_0, x_1 = self.backbone(batch.x_0, batch.incidence_hyperedges)
        model_out["x_0"] = x_0
        model_out["hyperedge"] = x_1
        model_out["batch"] = batch.batch
        return model_out


class SANWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within network"""

    def __init__(self, backbone):
        super().__init__(backbone)

    def __call__(self, batch):
        """Define logic for forward pass"""
        model_out = {"labels": batch.y, "batch": batch.batch}

        x_1 = self.backbone(batch.x_1, batch.up_laplacian_1, batch.down_laplacian_1)
        # Project the edge-level output of the model back to the node-level
        x_0 = torch.sparse.mm(batch.incidence_1, x_1)
        model_out["x_0"] = x_0
        return model_out





class SCNWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within network"""

    def __init__(self, backbone, **kwargs):
        super().__init__(backbone)
        if kwargs.get("out_channels", False):
            self.agg_conv_1 = topomodelx.base.conv.Conv(
                kwargs["out_channels"], kwargs["out_channels"], aggr_norm=False
            )
            self.agg_conv_2 = topomodelx.base.conv.Conv(
                kwargs["out_channels"], kwargs["out_channels"], aggr_norm=False
            )

            self.norm_1 = GraphNorm(kwargs["out_channels"])
            self.norm_2 = GraphNorm(kwargs["out_channels"])

        self.wrapper_readout = kwargs.get("wrapper_readout")

    def __call__(self, batch):
        """Define logic for forward pass"""
        model_out = {"labels": batch.y, "batch": batch.batch}
        
        
        def normalize_matrix(matrix):
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
            return normalized_matrix #torch.sparse_coo_tensor(
               # normalized_matrix.nonzero().T, normalized_matrix[normalized_matrix != 0], (n, n)
            #)
        
        laplacian_0 = normalize_matrix(batch.hodge_laplacian_0)
        laplacian_1 = normalize_matrix(batch.hodge_laplacian_1)
        laplacian_2 = normalize_matrix(batch.hodge_laplacian_2)
        x_0, x_1, x_2 = self.backbone(
            batch.x_0, batch.x_1, batch.x_2, laplacian_0, laplacian_1, laplacian_2
        )

        # Propagate signal down
        model_out["x_2"] = x_2
        model_out["x_1"] = self.norm_1(
            x_1 + self.agg_conv_1(model_out["x_2"], batch.incidence_2),
            batch.batch_1,
        )
        model_out["x_0"] = self.norm_2(
            x_0 + self.agg_conv_2(model_out["x_1"], batch.incidence_1), batch.batch
        )
        
        return model_out


class SCCNNWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within network"""

    def __init__(self, backbone, **kwargs):
        super().__init__(backbone)
        if kwargs.get("out_channels", False):
            self.agg_conv_1 = topomodelx.base.conv.Conv(
                kwargs["out_channels"], kwargs["out_channels"], aggr_norm=False
            )
            self.agg_conv_2 = topomodelx.base.conv.Conv(
                kwargs["out_channels"], kwargs["out_channels"], aggr_norm=False
            )

            self.norm_1 = GraphNorm(kwargs["out_channels"])
            self.norm_2 = GraphNorm(kwargs["out_channels"])

        self.wrapper_readout = kwargs.get("wrapper_readout")

    def __call__(self, batch):
        """Define logic for forward pass"""
        model_out = {"labels": batch.y, "batch": batch.batch}
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

        if self.wrapper_readout == "original":
            model_out["x_2"] = x_2
            model_out["x_1"] = x_1
            model_out["x_0"] = x_0

        elif self.wrapper_readout == "signal_prop_down":
            model_out["x_2"] = x_2
            model_out["x_1"] = self.norm_1(
                x_1 + self.agg_conv_1(model_out["x_2"], batch.incidence_2),
                batch.batch_1,
            )
            model_out["x_0"] = self.norm_2(
                x_0 + self.agg_conv_2(model_out["x_1"], batch.incidence_1), batch.batch
            )

        else:
            raise ValueError(f"Invalid wrapper readout method: {self.wrapper_readout}")

        return model_out


class SCCNWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within network"""

    def __init__(self, backbone, **kwargs):
        super().__init__(backbone)
        if kwargs.get("out_channels", False):
            self.agg_conv_1 = topomodelx.base.conv.Conv(
                kwargs["out_channels"], kwargs["out_channels"], aggr_norm=False
            )
            self.agg_conv_2 = topomodelx.base.conv.Conv(
                kwargs["out_channels"], kwargs["out_channels"], aggr_norm=False
            )

            self.norm_1 = GraphNorm(kwargs["out_channels"])
            self.norm_2 = GraphNorm(kwargs["out_channels"])

        self.wrapper_readout = kwargs.get("wrapper_readout")

    def __call__(self, batch):
        """Define logic for forward pass"""
        model_out = {"labels": batch.y, "batch": batch.batch}
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
        if len(output) == 3:
            x_0, x_1, x_2 = output["rank_0"], output["rank_1"], output["rank_2"]
            if self.wrapper_readout == "original":
                model_out["x_2"] = x_2
                model_out["x_1"] = x_1
                model_out["x_0"] = x_0

            elif self.wrapper_readout == "signal_prop_down":
                model_out["x_2"] = x_2
                model_out["x_1"] = self.norm_1(
                    x_1 + self.agg_conv_1(model_out["x_2"], batch.incidence_2),
                    batch.batch_1,
                )
                model_out["x_0"] = self.norm_2(
                    x_0 + self.agg_conv_2(model_out["x_1"], batch.incidence_1),
                    batch.batch,
                )

            else:
                raise ValueError(
                    f"Invalid wrapper readout method: {self.wrapper_readout}"
                )

        elif len(output) == 2:
            x_0, x_1 = output["rank_0"], output["rank_1"]
            if self.wrapper_readout == "original":
                model_out["x_1"] = x_1
                model_out["x_0"] = x_0

            elif self.wrapper_readout == "signal_prop_down":
                model_out["x_1"] = x_1
                model_out["x_0"] = self.norm_2(
                    x_0 + self.agg_conv_2(model_out["x_1"], batch.incidence_1),
                    batch.batch,
                )

            else:
                raise ValueError(
                    f"Invalid wrapper readout method: {self.wrapper_readout}"
                )

        else:
            raise ValueError(f"Invalid number of output tensors: {len(output)}")

        # for r in range(self.backbone.layers[0].max_rank):
        #     model_out[f"x_{r}"] = output[f"rank_{r}"]
        return model_out


class CANWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within network"""

    def __init__(self, backbone):
        super().__init__(backbone)

    def __call__(self, batch):
        """Define logic for forward pass"""
        model_out = {"labels": batch.y, "batch": batch.batch}
        x_1 = self.backbone(
            batch.x,
            batch.x_1,
            batch.adjacency_0.coalesce(),
            batch.down_laplacian_1.coalesce(),
            batch.up_laplacian_1.coalesce(),
        )
        x_0 = torch.sparse.mm(batch.incidence_1, x_1)

        model_out["x_1"] = x_1
        model_out["x_0"] = x_0
        return model_out


class CWNDCMWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within network"""

    def __init__(self, backbone):
        super().__init__(backbone)

    def __call__(self, batch):
        """Define logic for forward pass"""
        model_out = {"labels": batch.y, "batch": batch.batch}
        x_1 = self.backbone(
            batch.x_1,
            batch.down_laplacian_1.coalesce(),
            batch.up_laplacian_1.coalesce(),
        )
        x_0 = torch.sparse.mm(batch.incidence_1, x_1)

        model_out["x_1"] = x_1
        model_out["x_0"] = x_0
        return model_out


class CWNWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within network"""

    def __init__(self, backbone):
        super().__init__(backbone)

    def __call__(self, batch):
        """Define logic for forward pass"""
        model_out = {"labels": batch.y, "batch": batch.batch}
        x_0, x_1, x_2 = self.backbone(
            x_0=batch.x_0,
            x_1=batch.x_1,
            x_2=batch.x_2,
            neighborhood_0_to_1=batch.incidence_1.T,
            neighborhood_1_to_1=batch.adjacency_1,
            neighborhood_2_to_1=batch.incidence_2,
        )

        model_out["x_0"] = torch.mm(
            batch.incidence_1, x_1
        )  # + torch.mm(batch.incidence_1,torch.mm(batch.incidence_2, x_2))
        model_out["x_1"] = x_1
        model_out["x_2"] = x_2
        return model_out


class CCXNWrapper(DefaultWrapper):
    """Abstract class that provides an interface to loss logic within network"""

    def __init__(self, backbone):
        super().__init__(backbone)

    def __call__(self, batch):
        """Define logic for forward pass"""
        model_out = {"labels": batch.y, "batch": batch.batch}
        x_0, x_1, x_2 = self.backbone(
            x_0=batch.x_0,
            x_1=batch.x_1,
            neighborhood_0_to_0=batch.adjacency_0,
            neighborhood_1_to_2=batch.incidence_2.T,
        )
        model_out["x_0"] = x_0
        model_out["x_1"] = x_1
        model_out["x_2"] = x_2
        return model_out
