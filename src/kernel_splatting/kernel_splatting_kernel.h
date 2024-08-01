// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torch/torch.h>

torch::Tensor kernel_splatting_cuda(const torch::Tensor& input, const torch::Tensor& parameter, int64_t kernel_type);

std::tuple<torch::Tensor, torch::Tensor> kernel_splatting_cuda_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& parameter,
    int64_t kernel_type,
    bool input_requires_grad,
    bool parameter_requires_grad);
