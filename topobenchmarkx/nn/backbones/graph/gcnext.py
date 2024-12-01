"""Class implementing the GCNext model."""

import torch
from torch import nn
import numpy as np

from einops.layers.torch import Rearrange


class GCNext(nn.Module):
    """Blah.

    Blah.

    Parameters
    ----------
    config : dict
        Blah.
    dyna_idx : idk
        Idk.
    """

    def __init__(self, config, dyna_idx=None):
        self.config = config
        super().__init__()
        seq = 50  # love hard coding

        self.arr0 = Rearrange("b n d -> b d n")
        self.arr1 = Rearrange("b d n -> b n d")

        self.dynamic_layers = build_dynamic_layers(self.config.motion_mlp)

        self.temporal_fc_in = False  # config.motion_fc_in.temporal_fc
        self.temporal_fc_out = False  # config.motion_fc_out.temporal_fc
        if self.temporal_fc_in:
            self.motion_fc_in = nn.Linear(
                self.config.motion.h36m_input_length_dct,
                self.config.motion.h36m_input_length_dct,
            )
        else:
            self.motion_fc_in = nn.Linear(
                self.config.motion.dim, self.config.motion.dim
            )  # nn.Linear(66,66)
            self.in_weight = nn.Parameter(torch.eye(50, 50))
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(
                self.config.motion.h36m_input_length_dct,
                self.config.motion.h36m_input_length_dct,
            )
        else:
            self.motion_fc_out = nn.Linear(
                self.config.motion.dim, self.config.motion.dim
            )  # nn.Linear(66,66)
            self.out_weight = nn.Parameter(torch.eye(50, 50))

        self.reset_parameters()

        self.mlp = nn.Parameter(torch.empty(50, 4))
        nn.init.xavier_uniform_(self.mlp, gain=1e-8)
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
        # print("HELLO??!???!")
        tau = 1

        num_nodes = 50 * 22 * 3
        batch_size = x.size(0) // num_nodes

        # Reshape to original expected format [batch_size, num_nodes, sequence_length]
        motion_input = x.view(batch_size, 50, 66)

        # Create sparse adjacency matrix for use in GCBlock
        N = batch_size * num_nodes
        self.skl_mask = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1), device=edge_index.device),
            (N, N),
        ).to(self.mlp.device)

        # Original processing
        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(motion_input)
            motion_feats = self.arr0(motion_feats)
            motion_feats = torch.einsum(
                "bvt,tj->bvj", motion_feats, self.in_weight
            )

        # Process through dynamic layers
        # for i in range(len(self.dynamic_layers.layers)):
        #     if_make_dynamic = self.dyna_idx[0] <= i <= self.dyna_idx[1]
        #     motion_feats = self.dynamic_layers.layers[i](motion_feats, self.mlp, if_make_dynamic, tau)

        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = torch.einsum(
                "btv,tj->bjv", motion_feats, self.out_weight
            )

        # Reshape back to PyG format [batch*num_nodes, features]
        motion_feats = motion_feats.reshape(-1, 1)

        return motion_feats

        # # x is shape [batch*frames*joints*channels, 1]
        # # this is allegedly fine because edge_index only
        # # creates nodes between edges within the same graph

        # # Use edge_index directly instead of converting to dense matrix

        # if self.temporal_fc_in:
        #     motion_feats = self.arr0(x)
        #     motion_feats = self.motion_fc_in(motion_feats)
        # else:
        #     motion_feats = self.motion_fc_in(x)
        #     motion_feats = self.arr0(motion_feats)
        #     motion_feats = torch.einsum('bvt,tj->bvj', motion_feats, self.in_weight)

        # # Create sparse mask from edge_index
        # N = x.size(1)  # number of nodes
        # self.skl_mask = torch.sparse_coo_tensor(
        #     edge_index,
        #     torch.ones(edge_index.size(1), device=edge_index.device),
        #     (N, N)
        # ).to(self.mlp.device)

        # for i in range(len(self.dynamic_layers.layers)):
        #     if_make_dynamic = self.dyna_idx[0] <= i <= self.dyna_idx[1]
        #     motion_feats = self.dynamic_layers.layers[i](motion_feats, self.mlp, if_make_dynamic, tau)

        # # Process through dynamic layers
        # for i in range(len(self.dynamic_layers.layers)):
        #     if_make_dynamic = self.dyna_idx[0] <= i <= self.dyna_idx[1]
        #     motion_feats = self.dynamic_layers.layers[i](motion_feats, self.mlp, if_make_dynamic, tau=1)

        # if self.temporal_fc_out:
        #     motion_feats = self.motion_fc_out(motion_feats)
        #     motion_feats = self.arr1(motion_feats)
        # else:
        #     motion_feats = self.arr1(motion_feats)
        #     motion_feats = self.motion_fc_out(motion_feats)
        #     motion_feats = torch.einsum("btv,tj->bjv", motion_feats, self.out_weight)

        # # Remove batch dimension if it was added
        # if len(x.shape) == 2:
        #     motion_feats = motion_feats.squeeze(0)

        # return motion_feats


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


# class GCBlock(nn.Module):

#     def __init__(self, dim, seq, use_norm=True, use_spatial_fc=False, layernorm_axis='spatial'):
#         super().__init__()

#         if not use_spatial_fc:

#             # define update step
#             self.update = Temporal_FC(seq)

#             # define adjacency and mask for aggregation
#             skl = Skeleton(skl_type='h36m', joint_n=22).skeleton
#             skl = torch.tensor(skl, dtype=torch.float32, requires_grad=False)
#             bi_skl = torch.zeros(22, 22, requires_grad=False)
#             bi_skl[skl != 0] = 1.
#             self.skl_mask = bi_skl.cuda()

#             self.adj_j = nn.Parameter(torch.eye(22, 22))

#             self.traj_mask = (torch.tril(torch.ones(seq, seq, requires_grad=False), 1) * torch.triu(
#                 torch.ones(seq, seq, requires_grad=False), -1)).cuda()  # 三对角矩阵
#             for j in range(seq):
#                 self.traj_mask[j, j] = 0.
#             self.adj_t = nn.Parameter(torch.zeros(seq, seq))

#             self.adj_jc = nn.Parameter(torch.zeros(22, 3, 3))

#             self.adj_tj = nn.Parameter(torch.zeros(dim, seq, seq))


#         else:
#             self.update = Spatial_FC(dim)


#         if use_norm:
#             if layernorm_axis == 'spatial':
#                 self.norm0 = LN(dim)
#             elif layernorm_axis == 'temporal':
#                 self.norm0 = LN_v2(seq)
#             elif layernorm_axis == 'all':
#                 self.norm0 = nn.LayerNorm([dim, seq])
#             else:
#                 raise NotImplementedError
#         else:
#             self.norm0 = nn.Identity()

#         self.reset_parameters()


#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.update.fc.weight, gain=1e-8)

#         nn.init.constant_(self.update.fc.bias, 0)

#     def forward(self, x, mlp, if_make_dynamic, tau):
#         # 输入[bs,66,50]

#         b, v, t = x.shape
#         x1 = x.reshape(b, v//3, 3, t)
#         skl_mask = self.skl_mask
#         x1 = torch.einsum('vj,bjct->bvct', self.adj_j.mul(skl_mask), x1)
#         x1 = x1.reshape(b, v, t)

#         traj_mask = self.traj_mask
#         x2 = torch.einsum('ft,bnt->bnf', self.adj_t.mul(traj_mask), x)

#         x3 = x.reshape(b, v//3, 3, t)
#         x3 = torch.einsum('jkc,bjct->bjkt', self.adj_jc, x3)
#         x3 = x3.reshape(b, v, t)


#         x4 = torch.einsum('nft,bnt->bnf', self.adj_tj.mul(traj_mask.unsqueeze(0)), x)


#         prob = torch.einsum('bj,jk->bk', x.mean(1), mlp)  # [bs,50]->[bs,4]
#         if if_make_dynamic:
#             gate = nn.functional.gumbel_softmax(prob, tau=tau, hard=True)
#         else:
#             gate = torch.tensor([1., 0., 0., 0.]).unsqueeze(0).expand(x.shape[0], -1).cuda()


#         x2 = x2.unsqueeze(1)    # [bs,1,66,50]
#         x3 = x3.unsqueeze(1)
#         x4 = x4.unsqueeze(1)
#         x_opts = torch.cat([torch.zeros_like(x1).cuda().unsqueeze(1), x2, x3, x4], dim=1)   # [bs,4,66,50]


#         x_ = torch.einsum('bj,bjvt->bvt', gate, x_opts)


#         x_ = self.update(x1 + x_)
#         x_ = self.norm0(x_)
#         x = x + x_

#         return x


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
