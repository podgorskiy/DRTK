# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th

from drtk import kernel_splatting_ext

th.ops.load_library(kernel_splatting_ext.__file__)


@th.compiler.disable
def kernel_splatting(
        input: th.Tensor,
        parameter: th.Tensor,
        kernel_type: str = "disk",
        mode: str = "bilinear",
) -> th.Tensor:
    if mode != "bilinear" and mode != "nearest":
        raise ValueError(
                "kernel_splatting(): only 'bilinear' and 'nearest' modes are supported "
                "but got: '{}'".format(mode)
        )

    if mode == "bilinear":
        mode_enum = 0
    elif mode == "nearest":
        mode_enum = 1
    else:  # mode == 'bicubic'
        mode_enum = 2

    if kernel_type != "disk" and kernel_type != "gaussian":
        raise ValueError(
                "kernel_splatting(): only 'disk' and 'gaussian' kernel types are supported "
                "but got: '{}'".format(kernel_type)
        )

    if kernel_type == "disk":
        kernel_type_enum = 0
    elif kernel_type == "gaussian":
        kernel_type_enum = 1
    else:  # invalid
        kernel_type_enum = 3

    return th.ops.kernel_splatting_ext.kernel_splatting(
            input,
            parameter,
            kernel_type_enum,
    )
