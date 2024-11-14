"""Graph MLP loss function."""

import torch


def graph_mlp_contrast_loss(x_dis, adj_label, tau=1):
    """Graph MLP contrastive loss.

    Parameters
    ----------
    x_dis : torch.Tensor
        Distance matrix.
    adj_label : torch.Tensor
        Adjacency matrix.
    tau : float, optional
        Temperature parameter (default: 1).

    Returns
    -------
    torch.Tensor
        Contrastive loss.
    """
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1)) + 1e-8).mean()
    return loss
