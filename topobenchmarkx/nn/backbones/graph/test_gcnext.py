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

BATCH_SIZE = 4
N_JOINTS = 22
N_CHANNELS = 3
N_NODES_PER_FRAME = N_JOINTS * N_CHANNELS
N_FRAMES = 50


def test_temporal_joint_convolution_equivalence():
    """Test that TemporalJointConvolution forward pass matches reference implementation."""
    # Create random input tensor
    x = torch.randn(BATCH_SIZE, N_JOINTS, N_CHANNELS, N_FRAMES)

    # Current implementation output
    model = TemporalJointConvolution(
        n_joints=N_JOINTS, n_channels=N_CHANNELS, n_frames=N_FRAMES
    )
    output_current = model(x)

    # Reference implementation
    traj_mask = torch.tril(
        torch.ones(N_FRAMES, N_FRAMES, requires_grad=False), 1
    ) * torch.triu(torch.ones(N_FRAMES, N_FRAMES, requires_grad=False), -1)
    traj_mask.diagonal().fill_(0)

    # Input and output shapes differ, check to make sure that it's fine.
    reshaped_x = x.reshape(BATCH_SIZE, N_NODES_PER_FRAME, N_FRAMES)
    output_reference = torch.einsum(
        "nft,bnt->bnf", model.weights.mul(traj_mask.unsqueeze(0)), reshaped_x
    )
    reshaped_output_reference = output_reference.reshape(
        BATCH_SIZE, N_JOINTS, N_CHANNELS, N_FRAMES
    )

    # Check outputs match
    assert torch.allclose(
        output_current, reshaped_output_reference, rtol=1e-5, atol=1e-5
    ), "Current and reference implementations produce different outputs"


def test_joint_coordinate_convolution_equivalence():
    """Test that JointCoordinateConvolution forward pass matches reference implementation."""
    # Create random input tensor
    x = torch.randn(BATCH_SIZE, N_JOINTS, N_CHANNELS, N_FRAMES)

    # Current implementation output
    model = JointCoordinateConvolution()
    output_current = model(x)

    # Reference implementation
    reshaped_x = x.reshape(BATCH_SIZE, N_NODES_PER_FRAME, N_FRAMES)
    b, v, t = reshaped_x.shape
    x3 = reshaped_x.reshape(b, v // 3, 3, t)
    x3 = torch.einsum("jkc,bjct->bjkt", model.weights, x3)
    output_reference = x3.reshape(b, v, t)
    reshaped_output_reference = output_reference.reshape(
        BATCH_SIZE, N_JOINTS, N_CHANNELS, N_FRAMES
    )

    # Check outputs match
    assert torch.allclose(
        output_current, reshaped_output_reference, rtol=1e-5, atol=1e-5
    ), "Current and reference implementations produce different outputs"


def test_temporal_convolution_equivalence():
    """Test that TemporalConvolution forward pass matches reference implementation."""
    # Create random input tensor
    x = torch.randn(BATCH_SIZE, N_JOINTS, N_CHANNELS, N_FRAMES)

    # Current implementation output
    model = TemporalConvolution(n_frames=N_FRAMES)
    output_current = model(x)

    # Reference implementation
    reshaped_x = x.reshape(BATCH_SIZE, N_NODES_PER_FRAME, N_FRAMES)
    traj_mask = torch.tril(
        torch.ones(N_FRAMES, N_FRAMES, requires_grad=False), 1
    ) * torch.triu(torch.ones(N_FRAMES, N_FRAMES, requires_grad=False), -1)
    output_reference = torch.einsum(
        "ft,bnt->bnf", model.weights.mul(traj_mask), reshaped_x
    )
    reshaped_output_reference = output_reference.reshape(
        BATCH_SIZE, N_JOINTS, N_CHANNELS, N_FRAMES
    )

    # Check outputs match
    assert torch.allclose(
        output_current, reshaped_output_reference, rtol=1e-5, atol=1e-5
    ), "Current and reference implementations produce different outputs"


def test_skeletal_convolution_equivalence():
    """Test that SkeletalConvolution forward pass matches reference implementation."""
    # Create random input tensor
    x = torch.randn(BATCH_SIZE, N_JOINTS, N_CHANNELS, N_FRAMES)

    # Current implementation output
    model = SkeletalConvolution()
    output_current = model(x)

    ## COPIED FROM GCNEXT ##
    skl = H36MSkeleton().skl_mask.clone().detach()
    bi_skl = torch.zeros(22, 22, requires_grad=False)
    bi_skl[skl != 0] = 1.0
    skl_mask = bi_skl

    # Reference implementation
    reshaped_x = x.reshape(BATCH_SIZE, N_NODES_PER_FRAME, N_FRAMES)
    b, v, t = reshaped_x.shape
    x1 = reshaped_x.reshape(b, v // 3, 3, t)
    x1 = torch.einsum("vj,bjct->bvct", model.weights.mul(skl_mask), x1)
    output_reference = x1.reshape(b, v, t)
    reshaped_output_reference = output_reference.reshape(
        BATCH_SIZE, N_JOINTS, N_CHANNELS, N_FRAMES
    )

    # Check outputs match
    assert torch.allclose(
        output_current, reshaped_output_reference, rtol=1e-5, atol=1e-5
    ), "Current and reference implementations produce different outputs"
