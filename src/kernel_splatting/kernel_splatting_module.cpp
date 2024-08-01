// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/script.h>

#include <ATen/autocast_mode.h>

#ifndef NO_PYBIND
#include <torch/extension.h>
#endif

#include "kernel_splatting_kernel.h"

// Dispatch function
torch::Tensor kernel_splatting(const torch::Tensor& input, const torch::Tensor& parameter, int64_t kernel_type) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("kernel_splatting_ext::kernel_splatting", "")
                       .typed<decltype(kernel_splatting)>();
  return op.call(input, parameter, kernel_type);
}

// Ideally we would need to turn off autograd handling and re-dispatch, but we just call
// cuda kernels directly
class KernelSplattingFunction : public torch::autograd::Function<KernelSplattingFunction> {
 public:
  static torch::autograd::tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& input,
      const torch::Tensor& parameter,
      int64_t kernel_type) {
    ctx->set_materialize_grads(false);
    std::vector<torch::Tensor> save_list;
    save_list.push_back(input);
    save_list.push_back(parameter);
    ctx->save_for_backward(save_list);
    bool input_requires_grad = input.requires_grad();
    bool parameter_requires_grad = parameter.requires_grad();
    ctx->saved_data["data"] = std::make_tuple(input_requires_grad, parameter_requires_grad, kernel_type);
    auto out = kernel_splatting_cuda(input, parameter, kernel_type);
    return {out};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    bool input_requires_grad;
    bool parameter_requires_grad;
    int64_t kernel_type;
    std::tie(input_requires_grad, parameter_requires_grad, kernel_type) =
        ctx->saved_data["data"].to<std::tuple<bool, bool, int64_t>>();
    torch::autograd::tensor_list out;
    torch::Tensor grad_output = grad_outputs[0];
    if ((!input_requires_grad && !parameter_requires_grad) || !grad_output.defined()) {
      out.resize(3);
      return out;
    }
    const auto saved = ctx->get_saved_variables();
    const torch::Tensor& input = saved[0];
    const torch::Tensor& parameter = saved[1];
    auto grad_out = kernel_splatting_cuda_backward(
        grad_output, input, parameter, kernel_type, input_requires_grad, parameter_requires_grad);
    return {std::get<0>(grad_out), std::get<1>(grad_out), torch::Tensor()};
  }
};

torch::Tensor
kernel_splatting_autograd(const torch::Tensor& input, const torch::Tensor& parameter, int64_t kernel_type) {
  return KernelSplattingFunction::apply(input, parameter, kernel_type)[0];
}

torch::Tensor
kernel_splatting_autocast(const torch::Tensor& input, const torch::Tensor& parameter, int64_t kernel_type) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return kernel_splatting(
      at::autocast::cached_cast(torch::kFloat32, input),
      at::autocast::cached_cast(torch::kFloat32, parameter),
      kernel_type);
}

#ifndef NO_PYBIND
// Just so that we can import this file as a Python module to get the path and
// import the Torch ops.
PYBIND11_MODULE(kernel_splatting_ext, m) {}
#endif

TORCH_LIBRARY(kernel_splatting_ext, m) {
  m.def("kernel_splatting(Tensor input, Tensor parameter, int kernel_type) -> Tensor");
}

TORCH_LIBRARY_IMPL(kernel_splatting_ext, Autograd, m) {
  m.impl("kernel_splatting", &kernel_splatting_autograd);
}

TORCH_LIBRARY_IMPL(kernel_splatting_ext, Autocast, m) {
  m.impl("kernel_splatting", &kernel_splatting_autocast);
}

TORCH_LIBRARY_IMPL(kernel_splatting_ext, CUDA, m) {
  m.impl("kernel_splatting", &kernel_splatting_cuda);
}
