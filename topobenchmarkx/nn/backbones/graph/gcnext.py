"""Class implementing the GCNext model (https://ojs.aaai.org/index.php/AAAI/article/view/28375)."""

from typing import ClassVar

import numpy as np
import torch
from einops.layers.torch import Rearrange
from torch import nn


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

        self.dynamic_layers = TransGraphConvolution(config.motion_mlp)

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
            # initialised as identity, ie, no temporal relationships
            # then model can learn to mix info from diff frames

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
            Edge indices; this implementation assumes fully connected & doesn't care.

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

        # We don't care about edge_index at all.
        # This is sketchy, but fine.

        # Original processing

        # Step 1:
        # Processes the raw input motion sequence
        # Helps learn initial feature representations
        # Applied before the graph convolution layers

        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)
            # Transform across time, this means it's 50x50
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            # Transform across features (66×66) with temporal weighting
            # this means it's 66x66
            motion_feats = self.motion_fc_in(motion_input)
            motion_feats = self.arr0(motion_feats)  # now its (bs, 66, 50)
            motion_feats = torch.einsum(
                "bvt,tj->bvj", motion_feats, self.in_weight
            )  # now it;s [batch_size, 66, 50]

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
        super().__init__()
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
        self.skeletal_conv = SkeletalConvolution()
        self.temporal_conv = TemporalConvolution(n_frames=seq_len)
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
        x1 = self.skeletal_conv(x)
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
        # Kinda sketchily looks like they're always including x1...???
        # Biases model towards skeletal convolution
        x_opts = torch.stack([torch.zeros_like(x1), x2, x3, x4], dim=1)
        x_combined = torch.einsum("bj,bjvt->bvt", gate, x_opts)

        # Update and normalize
        x_out = self.update(x1 + x_combined)
        x_out = self.norm(x_out)

        # Residual connection
        return x + x_out


class TransGraphConvolution(nn.Module):
    """Blah.

    Blah.

    Parameters
    ----------
    config : dict
        Configuration containing:
        - dim: blaj
        - seq: blah
        - use_norm: blah
        - use_spatial_fc: blah
        - num_layers: blah
        - layernorm_axis: blah
    """

    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                GCBlock(
                    dim=config.hidden_dim,
                    seq_len=config.seq_len if "seq_len" in config else None,
                    use_norm=config.with_normalization,
                    use_spatial_fc=config.spatial_fc_only,
                    layernorm_axis=config.norm_axis,
                )
                for i in range(config.num_layers)
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


class SkeletalConvolution(nn.Module):
    """Convolution along skeletal connections between joints in H36MSkeleton."""

    def __init__(self):
        super().__init__()

        self.skl = H36MSkeleton()
        self.weights = nn.Parameter(
            torch.eye(self.skl.NUM_JOINTS, self.skl.NUM_JOINTS)
        )

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
        b, v, t = x.shape  # (batch, 66, 50)

        assert v == (self.skl.NUM_CHANNELS * self.skl.NUM_JOINTS)
        reshaped_x = x.reshape(
            b, self.skl.NUM_JOINTS, self.skl.NUM_CHANNELS, t
        )  # (batch, 22, 3, 50)

        masked_weights = self.weights.mul(self.skl.skl_mask)  # (22,22)
        processed_x = torch.einsum(
            "vj, bjct->bvct", masked_weights, reshaped_x
        )  # (22,22) x (batch,22,3,50) -> (batch,22,3,50)

        return processed_x.reshape(b, v, t)  # (batch, 66, 50)


class TemporalConvolution(nn.Module):
    """Convolution along temporal connections.

    Parameters
    ----------
    n_frames : int
        Number of frames, length of time dimension.
    """

    def __init__(self, n_frames: int):
        super().__init__()

        self.weights = nn.Parameter(torch.zeros(n_frames, n_frames))
        self.temporal_mask = self._create_tridiagonal_mask(n_frames)

    @staticmethod
    def _create_tridiagonal_mask(n_frames: int) -> torch.Tensor:
        """Create a tridiagonal mask matrix.

        Parameters
        ----------
        n_frames : int
            Number of frames, length of time dimension.

        Returns
        -------
        torch.tensor
            Temporal mask so you only look backwards in time, shape (num_frames, num_frames).
        """
        mask = torch.tril(torch.ones(n_frames, n_frames), 1) * torch.triu(
            torch.ones(n_frames, n_frames), -1
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
        # (n_frames, n_frames) x (batch, joint*channel, n_frames) -> (batch, joint*channel, n_frames)
        # (50, 50) x (256, 66, 50) -> (256, 66, 50)
        return torch.einsum(
            "ft,bnt->bnf", self.weights.mul(self.temporal_mask), x
        )


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


class H36MSkeleton:
    r"""Class for connections in Human3.6M Skeleton.

    Attributes
    ----------
    NUM_JOINTS (int): Number of joints in skeleton.
    NUM_CHANNELS (int): Number of channels per joint.
    USED_JOINT_INDICES (np.array[np.int64]): Numpy array containing relevant joint indices.
    """

    NUM_JOINTS: ClassVar = 22
    NUM_CHANNELS: ClassVar = 3

    USED_JOINT_INDICES: ClassVar = np.array(
        [
            2,
            3,
            4,
            5,
            7,
            8,
            9,
            10,
            12,
            13,
            14,
            15,
            17,
            18,
            19,
            21,
            22,
            25,
            26,
            27,
            29,
            30,
        ]
    ).astype(np.int64)

    def __init__(self):
        r"""H36M skeleton with 22 joints."""

        self.bone_list = self.generate_bone_list()
        self.skl_mask = self.generate_skl_mask()

    def compute_flat_index(self, t, j, c):
        r"""Compute flat index for motion matrix of shape (T,J,C).

        Parameters
        ----------
        t : int
            Time index in 3d matrix.
        j : int
            Joint index in 3d matrix.
        c : int
            Channel index in 3d matrix.

        Returns
        -------
        int
            Flat index in T*J*C vector.
        """
        return (
            t * self.NUM_JOINTS * self.NUM_CHANNELS + j * self.NUM_CHANNELS + c
        )

    def generate_bone_list(self):
        r"""Generate bones in H36M skeleton with 22 joints.

        Returns
        -------
        list[tup[int]]
            Edge list with bone links and self links.
        """
        self_links = [(i, i) for i in range(self.NUM_JOINTS)]
        joint_links = [
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

        return self_links + [(i - 1, j - 1) for (i, j) in joint_links]

    def generate_skl_mask(self):
        r"""Get skeleton mask for H36M skeleton with 22 joints.

        Returns
        -------
        list[tup[int]]
            Edge list with bone links and self links.
        """
        # Create adjacency matrix
        skl_mask = torch.zeros(
            self.NUM_JOINTS, self.NUM_JOINTS, requires_grad=False
        )
        for i, j in self.bone_list:
            skl_mask[i, j] = 1
            skl_mask[j, i] = 1
        return skl_mask

    # def generate_time_edges(self, n_times):
    #     r"""Generate list of edges only through time.

    #     Parameters
    #     ----------
    #     n_times : int
    #         Number of frames to consider.

    #     Returns
    #     -------
    #     torch.tensor
    #         Time edges.
    #     """
    #     time_edges = []
    #     for c in range(self.NUM_CHANNELS):
    #         for j in range(self.NUM_JOINTS):
    #             for t1 in range(n_times):
    #                 for t2 in range(n_times):
    #                     edge = [
    #                         self.compute_flat_index(t1, j, c),
    #                         self.compute_flat_index(t2, j, c),
    #                     ]
    #                     time_edges.append(edge)
    #     return torch.tensor(time_edges).T

    # def generate_bone_edges(self, n_times):
    #     """Generate list of edges along bones across all frames.

    #     Parameters
    #     ----------
    #     n_times : int
    #         Number of frames to consider.

    #     Returns
    #     -------
    #     torch.tensor
    #         Bone edges spanning all frames, shape [2, num_edges].
    #     """
    #     bone_edges = []

    #     # For each frame
    #     for t in range(n_times):
    #         # For each channel
    #         for c in range(self.NUM_CHANNELS):
    #             # For each bone connection
    #             for j1, j2 in self.bone_list:
    #                 edge = [
    #                     self.compute_flat_index(t, j1, c),
    #                     self.compute_flat_index(t, j2, c),
    #                 ]
    #                 bone_edges.append(edge)

    #     return torch.tensor(bone_edges).T
