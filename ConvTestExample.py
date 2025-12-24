# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map, HS, WS, BS
import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("enable_activation_reuse", [False])
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, groups, kernel_height, kernel_width, stride_h, stride_w, padding, dilation_h, dilation_w",
    [
        (1, 32, 128, 64, 64, 1, 3, 3, 1, 1, (1, 1, 1, 1), 1, 1),
        (1, 64, 128, 64, 64, 1, 3, 3, 1, 1, (1, 1, 1, 1), 1, 1),
        (1, 96, 128, 64, 64, 1, 3, 3, 1, 1, (1, 1, 1, 1), 1, 1),
        (1, 128, 128, 64, 64, 1, 3, 3, 1, 1, (1, 1, 1, 1), 1, 1),
        (1, 160, 128, 64, 64, 1, 3, 3, 1, 1, (1, 1, 1, 1), 1, 1),
        (1, 192, 128, 64, 64, 1, 3, 3, 1, 1, (1, 1, 1, 1), 1, 1),
        (1, 256, 128, 64, 64, 1, 3, 3, 1, 1, (1, 1, 1, 1), 1, 1),
        (1, 288, 128, 64, 64, 1, 3, 3, 1, 1, (1, 1, 1, 1), 1, 1),
        (1, 320, 128, 64, 64, 1, 3, 3, 1, 1, (1, 1, 1, 1), 1, 1),
        (1, 352, 128, 64, 64, 1, 3, 3, 1, 1, (1, 1, 1, 1), 1, 1),
        (1, 384, 128, 64, 64, 1, 3, 3, 1, 1, (1, 1, 1, 1), 1, 1),
    ],
)
@pytest.mark.parametrize("shard_layout", [HS])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4])
@pytest.mark.parametrize("has_bias", [True])
@pytest.mark.parametrize("fp32_accum", [False])
@pytest.mark.parametrize("packer_l1_acc", [False])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_conv_input_channels_sweep(
    device,
    torch_tensor_map,
    enable_activation_reuse,
    batch_size,
    input_channels,
    output_channels,
    input_height,
    input_width,
    groups,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding,
    dilation_h,
    dilation_w,
    shard_layout,
    output_dtype,
    input_dtype,
    weights_dtype,
    math_fidelity,
    has_bias,
    fp32_accum,
    packer_l1_acc,
    output_layout,
):
    config = None

    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        padding,
        config,
        shard_layout=shard_layout,
        output_layout=output_layout,
        has_bias=has_bias,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        run_twice=False,
        input_layout=ttnn.TILE_LAYOUT,
        input_dtype=input_dtype,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        groups=groups,
        enable_activation_reuse=enable_activation_reuse,
    )
