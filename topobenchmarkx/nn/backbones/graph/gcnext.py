"""Class implementing the GCNext model."""

import torch
from torch import nn
import numpy as np

from einops.layers.torch import Rearrange


class GCNext(nn.Module):
    """GCNext model from AAAI 2024 paper (https://ojs.aaai.org/index.php/AAAI/article/view/28375).

    This model processes motion sequences represented as graphs, where nodes represent
    joints and edges represent skeletal connections. It uses dynamic routing to learn
    different types of graph convolutions. Implementation has been minimally altered to handle sparse (edge_list) input
    and torch_geometric batching convention.

    Parameters
    ----------
    config : dict
        Configuration containing:
        - motion_dataset: Settings for input data dimensions
        - motion_mlp: Settings for the dynamic layers
    dyna_idx : tuple of int, optional
        Start and end indices for which layers should use dynamic routing.
        For example, (2, 4) means layers 2-4 will use dynamic routing.
    """

    def __init__(self, config, dyna_idx=None):
        super().__init__()

        self.config = config

        self.n_frames = config.motion_dataset.n_frames
        self.n_joints = config.motion_dataset.n_joints
        self.n_channels = config.motion_dataset.n_channels

        self.n_nodes_per_frame = self.n_joints * self.n_channels

        self.n_nodes = self.n_frames * self.n_joints * self.n_channels

        # seq = 50  # love hard coding

        self.arr0 = Rearrange("b n d -> b d n")
        self.arr1 = Rearrange("b d n -> b n d")

        self.dynamic_layers = build_dynamic_layers(config.motion_mlp)

        self.temporal_fc_in = False  # config.motion_fc_in.temporal_fc
        self.temporal_fc_out = False  # config.motion_fc_out.temporal_fc
        if self.temporal_fc_in:
            # if true then layer is 50x50
            self.motion_fc_in = nn.Linear(self.n_frames, self.n_frames)
        else:
            # else its 66x66
            self.motion_fc_in = nn.Linear(
                self.n_nodes_per_frame, self.n_nodes_per_frame
            )  # nn.Linear(66,66)
            # then this?
            self.in_weight = nn.Parameter(torch.eye(50, 50))

        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(
                self.n_frames,
                self.n_frames,
            )
        else:
            self.motion_fc_out = nn.Linear(
                self.n_nodes_per_frame, self.n_nodes_per_frame
            )  # nn.Linear(66,66)
            self.out_weight = nn.Parameter(torch.eye(50, 50))

        self.reset_parameters()

        # this is the layer that chooses which convolution
        self.mlp = nn.Parameter(torch.empty(50, 4))
        nn.init.xavier_uniform_(self.mlp, gain=1e-8)

        # idk this
        self.dyna_idx = dyna_idx

    def reset_parameters(self):
        """Idk."""
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, x, edge_index):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape [batch*frames*joints*channels, 1].
        edge_index : torch.Tensor
            Edge indices.

        Returns
        -------
        torch.Tensor
            Output node features.
        """
        tau = 1

        batch_size = x.size(0) // self.n_nodes

        # Reshape to expected format (256, 50, 66) (batch_size, n_frames, n_nodes)
        motion_input = x.view(
            batch_size, self.n_frames, self.n_nodes_per_frame
        )

        # Create sparse adjacency matrix for use in GCBlock
        N = batch_size * self.n_nodes
        self.skl_mask = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1), device=edge_index.device),
            (N, N),
        ).to(self.mlp.device)

        # Original processing
        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)
            # this means it's 50x50
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            # this means it's 66x66
            motion_feats = self.motion_fc_in(motion_input)
            motion_feats = self.arr0(motion_feats)
            motion_feats = torch.einsum(
                "bvt,tj->bvj", motion_feats, self.in_weight
            )

        # Process through dynamic layers
        for i in range(len(self.dynamic_layers.layers)):
            if_make_dynamic = self.dyna_idx[0] <= i <= self.dyna_idx[1]
            motion_feats = self.dynamic_layers.layers[i](
                motion_feats, self.mlp, if_make_dynamic, tau
            )

        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = torch.einsum(
                "btv,tj->bjv", motion_feats, self.out_weight
            )

        # Reshape back to torch_geometric format [batch*num_nodes, features]
        motion_feats = motion_feats.reshape(-1, 1)

        return motion_feats


class LN(nn.Module):
    """Blah.

    Blah.

    Parameters
    ----------
    dim : idk
        Idk.
    epsilon : idk
        Idk.
    """

    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape [batch*frames*joints*channels, 1].

        Returns
        -------
        torch.Tensor
            Output node features.
        """
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class LN_v2(nn.Module):
    """Blah v2.

    Blah.

    Parameters
    ----------
    dim : idk
        Idk.
    epsilon : idk
        Idk.
    """

    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape [batch*frames*joints*channels, 1].

        Returns
        -------
        torch.Tensor
            Output node features.
        """
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class Spatial_FC(nn.Module):
    """Blah.

    Blah.

    Parameters
    ----------
    dim : idk
        Idk.
    """

    def __init__(self, dim):
        super(Spatial_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.arr0 = Rearrange("b n d -> b d n")
        self.arr1 = Rearrange("b d n -> b n d")

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape [batch*frames*joints*channels, 1].

        Returns
        -------
        torch.Tensor
            Output node features.
        """
        x = self.arr0(x)
        x = self.fc(x)
        x = self.arr1(x)
        return x


class Temporal_FC(nn.Module):
    """Blah.

    Blah.

    Parameters
    ----------
    dim : idk
        Idk.
    """

    def __init__(self, dim):
        super(Temporal_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape [batch*frames*joints*channels, 1].

        Returns
        -------
        torch.Tensor
            Output node features.
        """

        x = self.fc(x)
        return x


class GCBlock(nn.Module):
    """Graph convolution block combining multiple types of convolutions.

    Parameters
    ----------
    dim : idk
        Idk.
    seq_len : idk
        Idk.
    use_norm : idk
        Idk.
    use_spatial_fc : idk
        Idk.
    layernorm_axis : idk
        Idk.
    """

    def __init__(
        self,
        dim: int,
        seq_len: int,
        use_norm: bool = True,
        use_spatial_fc: bool = False,
        layernorm_axis: str = "spatial",
    ):
        super().__init__()

        # Initialize different convolution types
        # self.skeletal_conv = SkeletalConvolution(dim, seq_len,)
        self.temporal_conv = TemporalConvolution(dim, seq_len)
        self.coordinate_conv = JointCoordinateConvolution(dim, seq_len)
        self.temp_joint_conv = TemporalJointConvolution(dim, seq_len)

        # Update and normalization layers
        self.update = (
            Spatial_FC(dim) if use_spatial_fc else Temporal_FC(seq_len)
        )
        self.norm = self._create_norm_layer(
            use_norm, layernorm_axis, dim, seq_len
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Blah."""
        nn.init.xavier_uniform_(self.update.fc.weight, gain=1e-8)
        nn.init.constant_(self.update.fc.bias, 0)

    def _create_norm_layer(
        self, use_norm: bool, axis: str, dim: int, seq_len: int
    ) -> nn.Module:
        """Create appropriate normalization layer.

        Parameters
        ----------
        use_norm : bool
            Idk.
        axis : bool
            Idk.
        dim : bool
            Idk.
        seq_len : bool
            Idk.

        Returns
        -------
        blah
            Blah.
        """
        if not use_norm:
            return nn.Identity()

        if axis == "spatial":
            return LN(dim)
        elif axis == "temporal":
            return LN_v2(seq_len)
        elif axis == "all":
            return nn.LayerNorm([dim, seq_len])
        else:
            raise ValueError(f"Unknown normalization axis: {axis}")

    def forward(
        self,
        x: torch.Tensor,
        mlp: torch.Tensor,
        if_make_dynamic: bool,
        tau: float,
    ) -> torch.Tensor:
        """Combine different graph convolutions using dynamic routing.

        Parameters
        ----------
        x : torch.Tensor [batch, vertices, time]
            Input features.
        mlp : torch.Tensor
            MLP weights for dynamic routing.
        if_make_dynamic : bool
            Whether to use dynamic routing.
        tau : float
            Temperature for Gumbel softmax.

        Returns
        -------
        torch.Tensor
            Updated node features.
        """
        # Apply different convolution types
        x1 = self.temporal_conv(x)  # self.skeletal_conv(x)#, self.skl_mask)
        x2 = self.temporal_conv(x)
        x3 = self.coordinate_conv(x)
        x4 = self.temp_joint_conv(x)

        # Dynamic routing
        prob = torch.einsum("bj,jk->bk", x.mean(1), mlp)
        gate = (
            nn.functional.gumbel_softmax(prob, tau=tau, hard=True)
            if if_make_dynamic
            else torch.tensor([1.0, 0.0, 0.0, 0.0])
            .unsqueeze(0)
            .expand(x.shape[0], -1)
            .to(x.device)
        )

        # Combine convolution results
        x_opts = torch.stack([torch.zeros_like(x1), x2, x3, x4], dim=1)
        x_combined = torch.einsum("bj,bjvt->bvt", gate, x_opts)

        # Update and normalize
        x_out = self.update(x1 + x_combined)
        x_out = self.norm(x_out)

        return x + x_out


class OldGCBlock(nn.Module):
    """Blah.

    Blah.

    Parameters
    ----------
    dim : idk
        Idk.
    seq : idk
        Idk.
    use_norm : idk
        Idk.
    use_spatial_fc : idk
        Idk.
    layernorm_axis : idk
        Idk.
    """

    def __init__(
        self,
        dim,
        seq,
        use_norm=True,
        use_spatial_fc=False,
        layernorm_axis="spatial",
    ):
        super().__init__()

        if not use_spatial_fc:
            # define update step
            self.update = Temporal_FC(seq)

            # Initialize learnable parameters
            self.adj_j = nn.Parameter(torch.eye(22, 22))

            self.traj_mask = (
                torch.tril(torch.ones(seq, seq, requires_grad=False), 1)
                * torch.triu(torch.ones(seq, seq, requires_grad=False), -1)
            ).cuda()  # tridiagonal matrix
            for j in range(seq):
                self.traj_mask[j, j] = 0.0
            self.adj_t = nn.Parameter(torch.zeros(seq, seq))

            self.adj_jc = nn.Parameter(torch.zeros(22, 3, 3))
            self.adj_tj = nn.Parameter(torch.zeros(dim, seq, seq))
        else:
            self.update = Spatial_FC(dim)

        # Normalization layers remain the same
        if use_norm:
            if layernorm_axis == "spatial":
                self.norm0 = LN(dim)
            elif layernorm_axis == "temporal":
                self.norm0 = LN_v2(seq)
            elif layernorm_axis == "all":
                self.norm0 = nn.LayerNorm([dim, seq])
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        """Blah."""
        nn.init.xavier_uniform_(self.update.fc.weight, gain=1e-8)

        nn.init.constant_(self.update.fc.bias, 0)

    def forward(self, x, mlp, if_make_dynamic, tau):
        """Much summary.

        Parameters
        ----------
        x : torch.Tensor
            Shape [batch_size, features, seq_len].
        mlp : torch.Tensor
            MLP weights.
        if_make_dynamic : bool
            Whether to use dynamic routing.
        tau : float
            Temperature for Gumbel softmax.

        Returns
        -------
        torch.Tensor
            Output node features.
        """
        b, v, t = x.shape
        x1 = x.reshape(b, v // 3, 3, t)

        # Use sparse matrix multiplication for skeletal connections
        if hasattr(self, "skl_mask") and isinstance(
            self.skl_mask, torch.Tensor
        ):
            if self.skl_mask.is_sparse:
                # Handle sparse skeletal mask
                x1_flat = x1.reshape(b, -1, t)  # [b, v, t]
                x1_flat = torch.sparse.mm(
                    self.skl_mask, x1_flat.reshape(-1, t)
                ).reshape(b, -1, t)
                x1 = x1_flat.reshape(b, v // 3, 3, t)
            else:
                # Fallback to dense multiplication if mask is dense
                x1 = torch.einsum(
                    "vj,bjct->bvct", self.adj_j.mul(self.skl_mask), x1
                )
        else:
            # Default behavior if no mask is provided
            x1 = torch.einsum("vj,bjct->bvct", self.adj_j, x1)

        x1 = x1.reshape(b, v, t)

        # Temporal connections remain the same
        traj_mask = self.traj_mask
        x2 = torch.einsum("ft,bnt->bnf", self.adj_t.mul(traj_mask), x)

        # Joint-coordinate connections
        x3 = x.reshape(b, v // 3, 3, t)
        x3 = torch.einsum("jkc,bjct->bjkt", self.adj_jc, x3)
        x3 = x3.reshape(b, v, t)

        # Temporal-joint connections
        x4 = torch.einsum(
            "nft,bnt->bnf", self.adj_tj.mul(traj_mask.unsqueeze(0)), x
        )

        # Dynamic routing
        prob = torch.einsum("bj,jk->bk", x.mean(1), mlp)
        if if_make_dynamic:
            gate = nn.functional.gumbel_softmax(prob, tau=tau, hard=True)
        else:
            gate = (
                torch.tensor([1.0, 0.0, 0.0, 0.0])
                .unsqueeze(0)
                .expand(x.shape[0], -1)
                .cuda()
            )

        # Combine different graph convolution results
        x2 = x2.unsqueeze(1)
        x3 = x3.unsqueeze(1)
        x4 = x4.unsqueeze(1)
        x_opts = torch.cat(
            [torch.zeros_like(x1).cuda().unsqueeze(1), x2, x3, x4], dim=1
        )

        x_ = torch.einsum("bj,bjvt->bvt", gate, x_opts)

        # Update and normalize
        x_ = self.update(x1 + x_)
        x_ = self.norm0(x_)
        x = x + x_

        return x


class TransGraphConvolution(nn.Module):
    """Blah.

    Blah.

    Parameters
    ----------
    dim : str
        Idk.
    seq : str
        Idk.
    use_norm : str
        Idk.
    use_spatial_fc : str
        Idk.
    num_layers : str
        Idk.
    layernorm_axis : str
        Idk.
    """

    def __init__(
        self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis
    ):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                GCBlock(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
                for i in range(num_layers)
            ]
        )

    def forward(self, x, mlp, if_make_dynamic, tau):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape [batch*frames*joints*channels, 1].
        mlp : str
            Blan.
        if_make_dynamic : str
            Bhal.
        tau : str
            Blah.

        Returns
        -------
        torch.Tensor
            Output node features.
        """
        x = self.layers(x, mlp, if_make_dynamic, tau)
        return x


def build_dynamic_layers(args):
    """Blah.

    Parameters
    ----------
    args : idk
        Blah.

    Returns
    -------
    TransGraphConvolution
        Blah.
    """
    if "seq_len" in args:
        seq_len = args.seq_len
    else:
        seq_len = None
    return TransGraphConvolution(
        dim=args.hidden_dim,
        seq=seq_len,
        use_norm=args.with_normalization,
        use_spatial_fc=args.spatial_fc_only,
        num_layers=args.num_layers,
        layernorm_axis=args.norm_axis,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string.

    Parameters
    ----------
    activation : str
        Blah.

    Returns
    -------
    Activation function
        Blah.
    """
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    if activation == "silu":
        return nn.SiLU
    # if activation == 'swish':
    #    return nn.Hardswish
    if activation == "softplus":
        return nn.Softplus
    if activation == "tanh":
        return nn.Tanh
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _get_norm_fn(norm):
    """Return a norm given a string.

    Parameters
    ----------
    norm : str
        Blah.

    Returns
    -------
    Norm
        Blah.
    """

    if norm == "batchnorm":
        return nn.BatchNorm1d
    if norm == "layernorm":
        return nn.LayerNorm
    if norm == "instancenorm":
        return nn.InstanceNorm1d
    raise RuntimeError(f"norm should be batchnorm/layernorm, not {norm}.")


class Skeleton:
    """Blah.

    Blah.

    Parameters
    ----------
    skl_type : idk
        Idk.
    joint_n : idk
        Idk.
    """

    def __init__(self, skl_type="h36m", joint_n=22):
        self.joint_n = joint_n
        self.get_bone(skl_type)
        self.get_skeleton()

    def get_bone(self, skl_type):
        """Blah.

        Parameters
        ----------
        skl_type : str
            Blah.
        """
        self_link = [(i, i) for i in range(self.joint_n)]
        if skl_type == "h36m":
            if self.joint_n == 22:
                joint_link_ = [
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (5, 6),
                    (6, 7),
                    (7, 8),
                    (1, 9),
                    (5, 9),
                    (9, 10),
                    (10, 11),
                    (11, 12),
                    (10, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                    (15, 17),
                    (10, 18),
                    (18, 19),
                    (19, 20),
                    (20, 21),
                    (20, 22),
                ]
            if self.joint_n == 17:
                joint_link_ = [
                    (1, 2),
                    (2, 3),
                    (4, 5),
                    (5, 6),
                    (1, 7),
                    (4, 7),
                    (7, 8),
                    (8, 9),
                    (8, 10),
                    (10, 11),
                    (11, 12),
                    (11, 13),
                    (8, 14),
                    (14, 15),
                    (15, 16),
                    (15, 17),
                ]
            if self.joint_n == 11:
                joint_link_ = [
                    (1, 2),
                    (3, 4),
                    (5, 1),
                    (5, 3),
                    (5, 6),
                    (6, 7),
                    (6, 8),
                    (8, 9),
                    (6, 10),
                    (10, 11),
                    (3, 10),
                    (1, 8),
                ]
            if self.joint_n == 9:
                joint_link_ = [
                    (1, 3),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (4, 6),
                    (6, 7),
                    (4, 8),
                    (8, 9),
                    (1, 8),
                    (2, 6),
                ]
            if self.joint_n == 6:
                joint_link_ = [
                    (1, 3),
                    (2, 3),
                    (3, 4),
                    (3, 5),
                    (3, 6),
                    (1, 6),
                    (2, 5),
                ]
            if self.joint_n == 2:
                joint_link_ = [(1, 2)]
        joint_link = [(i - 1, j - 1) for (i, j) in joint_link_]
        self.bone = self_link + joint_link

    def get_skeleton(self):
        """Blah."""
        skl = np.zeros((self.joint_n, self.joint_n))
        for i, j in self.bone:
            skl[j, i] = 1
            skl[i, j] = 1
        self.skeleton = skl


###############################################################
#### Different types of graph convolutions used in GCNext. ####
###############################################################


class BaseGraphConvolution(nn.Module):
    """Base class for all graph convolution types.

    Parameters
    ----------
    dim : str
        Blah.
    seq_len : str
        Blah.
    """

    def __init__(self, dim: int, seq_len: int):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skeletal graph convolution.

        Parameters
        ----------
        x : torch.Tensor [batch, vertices, time]
            Input features.

        Returns
        -------
        torch.Tensor [batch, vertices, time]
            Features after skeletal convolution.
        """
        raise NotImplementedError


class SkeletalConvolution(BaseGraphConvolution):
    """Convolution along skeletal connections between joints.

    Parameters
    ----------
    dim : str
        Blah.
    seq_len : str
        Blah.
    num_joints : str
        Blah.
    """

    def __init__(self, dim: int, seq_len: int, num_joints: int = 22):
        super().__init__(dim, seq_len)
        self.adj_j = nn.Parameter(torch.eye(num_joints, num_joints))

        batch_size = 256
        # Create batched edge indices
        skl = Skeleton()
        base_edge_index = torch.tensor(skl.bone).t()  # [2, num_edges]
        # num_nodes_per_graph = self.n_joints

        # # Create offsets for each graph in the batch
        # batch_offsets = torch.arange(
        #     0,
        #     batch_size * self.n_frames * num_nodes_per_graph,
        #     num_nodes_per_graph,
        #     # device=edge_index.device
        # )

        # # Repeat edge indices for each graph and add appropriate offsets
        # batched_edge_index = base_edge_index.unsqueeze(1).expand(-1, len(batch_offsets), -1)
        # batched_edge_index = batched_edge_index + batch_offsets.view(1, -1, 1)
        # batched_edge_index = batched_edge_index.permute(0, 2, 1).reshape(2, -1)

        # Create sparse adjacency matrix
        N = 256 * 50 * 22 * 3
        self.skl_mask = torch.sparse_coo_tensor(
            base_edge_index,
            torch.ones(base_edge_index.size(1), device=base_edge_index.device),
            (N, N),
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # , skl_mask: torch.Tensor
        """Apply skeletal graph convolution.

        Parameters
        ----------
        x : torch.Tensor [batch, vertices, time]
            Input features.

        Returns
        -------
        torch.Tensor [batch, vertices, time]
            Features after skeletal convolution.
        """
        self.skl_mask = self.skl_mask.to(x.device)

        b, v, t = x.shape
        x1 = x.reshape(b, v // 3, 3, t)

        if self.skl_mask.is_sparse:
            x1_flat = x1.reshape(b, -1, t)
            x1_flat = torch.sparse.mm(
                self.skl_mask.to(x.device), x1_flat.reshape(-1, t)
            ).reshape(b, -1, t)
            x1 = x1_flat.reshape(b, v // 3, 3, t)
        else:
            x1 = torch.einsum(
                "vj,bjct->bvct", self.adj_j.mul(self.skl_mask), x1
            )

        return x1.reshape(b, v, t)


class TemporalConvolution(BaseGraphConvolution):
    """Convolution along temporal connections.

    Parameters
    ----------
    dim : str
        Blah.
    seq_len : str
        Blah.
    """

    def __init__(self, dim: int, seq_len: int):
        super().__init__(dim, seq_len)
        self.adj_t = nn.Parameter(torch.zeros(seq_len, seq_len))
        # Create tridiagonal mask
        self.register_buffer(
            "traj_mask", self._create_tridiagonal_mask(seq_len)
        )

    @staticmethod
    def _create_tridiagonal_mask(seq_len: int) -> torch.Tensor:
        """Create a tridiagonal mask matrix.

        Parameters
        ----------
        seq_len : str
            Blah.

        Returns
        -------
        blah
            Blan.
        """
        mask = torch.tril(torch.ones(seq_len, seq_len), 1) * torch.triu(
            torch.ones(seq_len, seq_len), -1
        )
        mask.diagonal().fill_(0)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal graph convolution.

        Parameters
        ----------
        x : torch.Tensor [batch, vertices, time]
            Input features.

        Returns
        -------
        torch.Tensor [batch, vertices, time]
            Features after temporal convolution.
        """
        return torch.einsum("ft,bnt->bnf", self.adj_t.mul(self.traj_mask), x)


class JointCoordinateConvolution(BaseGraphConvolution):
    """Convolution between joint coordinates.

    Parameters
    ----------
    dim : str
        Blah.
    seq_len : str
        Blah.
    num_joints : int
        Bla.
    """

    def __init__(self, dim: int, seq_len: int, num_joints: int = 22):
        super().__init__(dim, seq_len)
        self.adj_jc = nn.Parameter(torch.zeros(num_joints, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply joint-coordinate graph convolution.

        Parameters
        ----------
        x : torch.Tensor [batch, vertices, time]
            Input features.

        Returns
        -------
        torch.Tensor [batch, vertices, time]
            Features after joint-coordinate convolution.
        """
        b, v, t = x.shape
        x3 = x.reshape(b, v // 3, 3, t)
        x3 = torch.einsum("jkc,bjct->bjkt", self.adj_jc, x3)
        return x3.reshape(b, v, t)


class TemporalJointConvolution(BaseGraphConvolution):
    """Convolution combining temporal and joint relationships.

    Parameters
    ----------
    dim : str
        Blah.
    seq_len : str
        Blah.
    """

    def __init__(self, dim: int, seq_len: int):
        super().__init__(dim, seq_len)
        self.adj_tj = nn.Parameter(torch.zeros(dim, seq_len, seq_len))
        self.register_buffer(
            "traj_mask", TemporalConvolution._create_tridiagonal_mask(seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal-joint graph convolution.

        Parameters
        ----------
        x : torch.Tensor [batch, vertices, time]
            Input features.

        Returns
        -------
        torch.Tensor [batch, vertices, time]
            Features after temporal-joint convolution.
        """
        return torch.einsum(
            "nft,bnt->bnf", self.adj_tj.mul(self.traj_mask.unsqueeze(0)), x
        )
