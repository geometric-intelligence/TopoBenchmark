"""Unit tests for gcnext."""

import numpy as np
import pytest
import torch

from topobenchmarkx.nn.backbones.graph.gcnext import (
    H36MSkeleton,
    JointCoordinateConvolution,
    SkeletalConvolution,
    TemporalConvolution,
    TemporalJointConvolution,
)


def test_temporal_joint_convolution_equivalence():
    """Test that TemporalJointConvolution forward pass matches reference implementation."""
    # Setup test parameters
    batch_size = 4
    num_joints = 22
    channels = 3
    seq_len = 50
    vertices = num_joints * channels
    dim = vertices

    # Create random input tensor
    x = torch.randn(batch_size, vertices, seq_len)

    # Current implementation output
    model = TemporalJointConvolution(n_nodes_per_frame=dim, n_frames=seq_len)
    output_current = model(x)

    # Reference implementation
    traj_mask = torch.tril(
        torch.ones(seq_len, seq_len, requires_grad=False), 1
    ) * torch.triu(torch.ones(seq_len, seq_len, requires_grad=False), -1)
    traj_mask.diagonal().fill_(0)

    output_reference = torch.einsum(
        "nft,bnt->bnf", model.weights.mul(traj_mask.unsqueeze(0)), x
    )

    # Check outputs match
    assert torch.allclose(
        output_current, output_reference, rtol=1e-5, atol=1e-5
    ), "Current and reference implementations produce different outputs"


def test_joint_coordinate_convolution_equivalence():
    """Test that JointCoordinateConvolution forward pass matches reference implementation."""
    # Setup test parameters
    batch_size = 4
    num_joints = 22
    channels = 3
    seq_len = 50
    vertices = num_joints * channels

    # Create random input tensor
    x = torch.randn(batch_size, vertices, seq_len)

    # Current implementation output
    model = JointCoordinateConvolution()
    output_current = model(x)

    # Reference implementation
    b, v, t = x.shape
    x3 = x.reshape(b, v // 3, 3, t)
    x3 = torch.einsum("jkc,bjct->bjkt", model.weights, x3)
    output_reference = x3.reshape(b, v, t)

    # Check outputs match
    assert torch.allclose(
        output_current, output_reference, rtol=1e-5, atol=1e-5
    ), "Current and reference implementations produce different outputs"


def test_temporal_convolution_equivalence():
    """Test that TemporalConvolution forward pass matches reference implementation."""
    # Setup test parameters
    batch_size = 4
    num_joints = 22
    channels = 3
    seq_len = 50
    vertices = num_joints * channels

    # Create random input tensor
    x = torch.randn(batch_size, vertices, seq_len)

    # Current implementation output
    model = TemporalConvolution(n_frames=seq_len)
    output_current = model(x)

    # Reference implementation
    traj_mask = torch.tril(
        torch.ones(seq_len, seq_len, requires_grad=False), 1
    ) * torch.triu(torch.ones(seq_len, seq_len, requires_grad=False), -1)
    output_reference = torch.einsum(
        "ft,bnt->bnf", model.weights.mul(traj_mask), x
    )

    # Check outputs match
    assert torch.allclose(
        output_current, output_reference, rtol=1e-5, atol=1e-5
    ), "Current and reference implementations produce different outputs"


def test_skeletal_convolution_equivalence():
    """Test that SkeletalConvolution forward pass matches reference implementation."""
    # Setup test parameters
    batch_size = 4
    num_joints = 22
    channels = 3
    seq_len = 50
    vertices = num_joints * channels

    # Create random input tensor
    x = torch.randn(batch_size, vertices, seq_len)

    # Current implementation output
    model = SkeletalConvolution()
    output_current = model(x)

    ## COPIED FROM GCNEXT ##
    skl = H36MSkeleton().skl_mask.clone().detach()
    bi_skl = torch.zeros(22, 22, requires_grad=False)
    bi_skl[skl != 0] = 1.0
    skl_mask = bi_skl

    # Reference implementation
    b, v, t = x.shape
    x1 = x.reshape(b, v // 3, 3, t)
    x1 = torch.einsum("vj,bjct->bvct", model.weights.mul(skl_mask), x1)
    output_reference = x1.reshape(b, v, t)

    # Check outputs match
    assert torch.allclose(
        output_current, output_reference, rtol=1e-5, atol=1e-5
    ), "Current and reference implementations produce different outputs"
