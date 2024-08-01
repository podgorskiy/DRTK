// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <c10/cuda/CUDAGuard.h>
#include <cuda_math_helper.h>
#include <torch/types.h>

#include <grid_utils.h>
#include <kernel_utils.h>
#include <tensor_list.h>

using namespace math;

enum class Kernel {
  Disk = 0,
  RadialGaussian = 1,
  Gaussian = 2,
};

using at::native::fastAtomicAdd;
using at::native::within_bounds_2d;

inline uint __device__ murmur_hash(uint data, uint seed) {
  const uint m = 0x5bd1e995u;
  uint h = seed;
  data *= m;
  data ^= data >> 24u;
  data *= m;
  h *= m;
  h ^= data;
  h ^= h >> 13u;
  h *= m;
  h ^= h >> 15u;
  return h;
}

inline float __device__ uint_to_uniform_0_1_float(uint src) {
  return __uint_as_float(src & 0x007fffffu | 0x3f800000u) - 1.0f;
}

inline float __device__ radical_inverse(uint bits) {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return float(bits) * 2.3283064365386963e-10f;
}

template <typename scalar_t, typename index_t>
__device__ void splat_bilinear(
    const scalar_t* __restrict input_ptr,
    scalar_t* __restrict output_ptr,
    scalar_t alpha,
    scalar_t ix,
    scalar_t iy,
    index_t input_sC,
    index_t output_N_offset, index_t output_sC, index_t output_sH, index_t output_sW, index_t C, index_t H, index_t W, index_t memory_span){
  // get NE, NW, SE, SW pixel values from (x, y)
  auto ix_nw = static_cast<index_t>(::floor(ix));
  auto iy_nw = static_cast<index_t>(::floor(iy));
  index_t ix_ne = ix_nw + 1;
  index_t iy_ne = iy_nw;
  index_t ix_sw = ix_nw;
  index_t iy_sw = iy_nw + 1;
  index_t ix_se = ix_nw + 1;
  index_t iy_se = iy_nw + 1;

  // get surfaces to each neighbor:
  scalar_t nw = (ix_se - ix) * (iy_se - iy);
  scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
  scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
  scalar_t se = (ix - ix_nw) * (iy - iy_nw);

  //  Add to output pixel bilinear weighted source pixel value
  for (index_t c = 0; c < C; ++c) {
    scalar_t v = input_ptr[c * input_sC] * alpha;
    safe_add_2d(output_ptr, iy_nw, ix_nw, output_sH, output_sW,  H, W, v * nw, output_N_offset + output_sC * c,  memory_span);
    safe_add_2d(output_ptr, iy_ne, ix_ne, output_sH, output_sW,  H, W, v * ne, output_N_offset + output_sC * c,  memory_span);
    safe_add_2d(output_ptr, iy_sw, ix_sw, output_sH, output_sW,  H, W, v * sw, output_N_offset + output_sC * c,  memory_span);
    safe_add_2d(output_ptr, iy_se, ix_se, output_sH, output_sW,  H, W, v * se, output_N_offset + output_sC * c,  memory_span);
  }
}

template <typename scalar_t, typename index_t>
__device__ TVec2<scalar_t> sample_bilinear_backward(
    const TensorInfoCompact<scalar_t, index_t, 4>& input,
    const TensorInfoCompact<scalar_t, index_t, 4>& grad_input,
    const TensorInfoCompact<scalar_t, index_t, 4>& grad_output,
    scalar_t x,
    scalar_t y,
    const index_t w,
    const index_t h,
    const index_t n,
    const index_t C,
    scalar_t alpha,
    const GridSamplerPadding padding_mode,
    bool align_corners,
    index_t grad_input_memory_span) {
  index_t gOut_sN = grad_output.strides[0];
  index_t gOut_sC = grad_output.strides[1];
  index_t gOut_sH = grad_output.strides[2];
  index_t gOut_sW = grad_output.strides[3];
  index_t gInp_sN = grad_input.strides[0];
  index_t gInp_sC = grad_input.strides[1];
  index_t gInp_sH = grad_input.strides[2];
  index_t gInp_sW = grad_input.strides[3];

  index_t inp_H = input.sizes[2];
  index_t inp_W = input.sizes[3];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sH = input.strides[2];
  index_t inp_sW = input.strides[3];

  // multipliers for gradients on ix and iy
  TVec2<scalar_t> gi_mult;
  scalar_t ix = grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &gi_mult.x);
  scalar_t iy = grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners, &gi_mult.y);

  // get NE, NW, SE, SW pixel values from (x, y)
  index_t ix_nw = static_cast<index_t>(::floor(ix));
  index_t iy_nw = static_cast<index_t>(::floor(iy));
  index_t ix_ne = ix_nw + 1;
  index_t iy_ne = iy_nw;
  index_t ix_sw = ix_nw;
  index_t iy_sw = iy_nw + 1;
  index_t ix_se = ix_nw + 1;
  index_t iy_se = iy_nw + 1;

  // get surfaces to each neighbor:
  scalar_t nw = (ix_se - ix) * (iy_se - iy);
  scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
  scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
  scalar_t se = (ix - ix_nw) * (iy - iy_nw);

  TVec2<scalar_t> gi = {scalar_t(0), scalar_t(0)};
  scalar_t* gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
  index_t NC_offset = n * gInp_sN;
  scalar_t* inp_ptr_NC = input.data + n * inp_sN;
  for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, NC_offset += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
    scalar_t gOut = *gOut_ptr_NCHW * alpha;

    // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
    safe_add_2d(
        grad_input.data, iy_nw, ix_nw, gInp_sH, gInp_sW, inp_H, inp_W, nw * gOut, NC_offset, grad_input_memory_span);
    safe_add_2d(
        grad_input.data, iy_ne, ix_ne, gInp_sH, gInp_sW, inp_H, inp_W, ne * gOut, NC_offset, grad_input_memory_span);
    safe_add_2d(
        grad_input.data, iy_sw, ix_sw, gInp_sH, gInp_sW, inp_H, inp_W, sw * gOut, NC_offset, grad_input_memory_span);
    safe_add_2d(
        grad_input.data, iy_se, ix_se, gInp_sH, gInp_sW, inp_H, inp_W, se * gOut, NC_offset, grad_input_memory_span);

    // calculate grad_grid
    if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
      scalar_t nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
      gi.x -= nw_val * (iy_se - iy) * gOut;
      gi.y -= nw_val * (ix_se - ix) * gOut;
    }
    if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
      scalar_t ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
      gi.x += ne_val * (iy_sw - iy) * gOut;
      gi.y -= ne_val * (ix - ix_sw) * gOut;
    }
    if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
      scalar_t sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
      gi.x -= sw_val * (iy - iy_ne) * gOut;
      gi.y += sw_val * (ix_ne - ix) * gOut;
    }
    if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
      scalar_t se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
      gi.x += se_val * (iy - iy_nw) * gOut;
      gi.y += se_val * (ix - ix_nw) * gOut;
    }
  }
  return gi_mult * gi;
}

template <typename scalar_t, typename index_t, Kernel kernel_type>
C10_LAUNCH_BOUNDS_1(256)
__global__ void kernel_splatting_kernel(
    const index_t nthreads,
    TensorInfo<scalar_t, index_t> input,
    TensorInfo<scalar_t, index_t> parameter,
    TensorInfo<scalar_t, index_t> output,
    index_t memory_span,
    uint seed) {
  index_t N = input.sizes[0];
  index_t C = input.sizes[1];
  index_t H = input.sizes[2];
  index_t W = input.sizes[3];
  index_t P = parameter.sizes[1];

  index_t input_sN = input.strides[0];
  index_t input_sC = input.strides[1];
  index_t input_sH = input.strides[2];
  index_t input_sW = input.strides[3];

  index_t parameter_sN = parameter.strides[0];
  index_t parameter_sP = parameter.strides[1];
  index_t parameter_sH = parameter.strides[2];
  index_t parameter_sW = parameter.strides[3];

  index_t output_sN = output.strides[0];
  index_t output_sC = output.strides[1];
  index_t output_sH = output.strides[2];
  index_t output_sW = output.strides[3];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % W;
    const index_t h = (index / W) % H;
    const index_t n = index / (H * W);

    const scalar_t* __restrict input_ptr = input.data + n * input_sN + h * input_sH + w * input_sW;
    const scalar_t* __restrict parameter_ptr = parameter.data + n * parameter_sN + h * parameter_sH + w * parameter_sW;
    scalar_t* __restrict output_ptr = output.data;

    if (kernel_type == Kernel::Disk) {
      const scalar_t r = *parameter_ptr;
      const int num_samples = min(100, 1 + int(scalar_t(3.14115) * r * r));

      for (int i = 0; i < num_samples; ++i) {
        const scalar_t sample_r = r * sqrt(scalar_t(i) / scalar_t(num_samples));
        const scalar_t sample_a = uint_to_uniform_0_1_float(murmur_hash(index, seed)) + radical_inverse(i);

        const scalar_t x = w + sample_r * cos(scalar_t(2) * scalar_t(3.1415) * sample_a);
        const scalar_t y = h + sample_r * sin(scalar_t(2) * scalar_t(3.1415) * sample_a);
        const scalar_t alpha = scalar_t(1) / scalar_t(num_samples);
        splat_bilinear(input_ptr, output_ptr, alpha, x, y, input_sC, n * output_sN, output_sC, output_sH, output_sW, C, H, W, memory_span);
      }
    }
  }
}

template <
    typename scalar_t,
    typename index_t,
    Kernel kernel_type,
    bool input_requires_grad,
    bool parameter_requires_grad>
C10_LAUNCH_BOUNDS_1(256)
__global__ void kernel_splatting_backward_kernel(
    const index_t nthreads,
    TensorInfo<scalar_t, index_t> grad_output,
    TensorInfo<scalar_t, index_t> input,
    TensorInfo<scalar_t, index_t> parameter,
    TensorInfo<scalar_t, index_t> grad_input,
    TensorInfo<scalar_t, index_t> grad_parameter) {
  //  index_t C = inputs[0].sizes[1];
  //  index_t inp_H = inputs[0].sizes[2];
  //  index_t inp_W = inputs[0].sizes[3];
  //  index_t out_H = grid.sizes[1];
  //  index_t out_W = grid.sizes[2];
  //  index_t grid_sN = grid.strides[0];
  //  index_t grid_sH = grid.strides[1];
  //  index_t grid_sW = grid.strides[2];
  //  index_t grid_sCoor = grid.strides[3];
  //
  //  index_t gGrid_sW = grad_grid.strides[2];
  //
  //  index_t vt_dxdy_img_sN = vt_dxdy_img.strides[0];
  //  index_t vt_dxdy_img_sH = vt_dxdy_img.strides[1];
  //  index_t vt_dxdy_img_sW = vt_dxdy_img.strides[2];
  //  index_t vt_dxdy_img_s3 = vt_dxdy_img.strides[3];
  //  index_t vt_dxdy_img_s4 = vt_dxdy_img.strides[4];
  //
  //  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
  //    const index_t w = index % out_W;
  //    const index_t h = (index / out_W) % out_H;
  //    const index_t n = index / (out_H * out_W);
  //    const auto grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;
  //    const index_t vt_dxdy_img_offset = n * vt_dxdy_img_sN + h * vt_dxdy_img_sH + w * vt_dxdy_img_sW;
  //
  //    // get the corresponding input x, y co-ordinates from grid
  //    scalar_t u = grid.data[grid_offset];
  //    scalar_t v = grid.data[grid_offset + grid_sCoor];
  //
  //    scalar_t dudx = vt_dxdy_img.data[vt_dxdy_img_offset];
  //    scalar_t dvdx = vt_dxdy_img.data[vt_dxdy_img_offset + vt_dxdy_img_s4];
  //
  //    scalar_t dudy = vt_dxdy_img.data[vt_dxdy_img_offset + vt_dxdy_img_s3];
  //    scalar_t dvdy = vt_dxdy_img.data[vt_dxdy_img_offset + vt_dxdy_img_s3 + vt_dxdy_img_s4];
  //
  //    scalar_t px = pow(pow(abs(dudx * inp_W), 2.0f) + pow(abs(dvdx * inp_H), 2.0f) + 1e-12f, 0.5f);
  //    scalar_t py = pow(pow(abs(dudy * inp_W), 2.0f) + pow(abs(dvdy * inp_H), 2.0f) + 1e-12f, 0.5f);
  //
  //    scalar_t p_max = max(px, py);
  //    scalar_t p_min = min(px, py);
  //
  //    // # See p.255 of OpenGL Core Profile
  //    // # N = min(ceil(Pmax/Pmin),maxAniso)
  //    scalar_t N = min(ceil(p_max / p_min), (scalar_t)max_aniso);
  //    if (p_min == 0.0 || N == 0) {
  //      N = 1;
  //    }
  //
  //    // Lambda' = log2(Pmax/N)
  //    scalar_t lambda_ = log2(p_max / N);
  //    if (isnan(lambda_) || isinf(lambda_)) {
  //      lambda_ = 0.0f;
  //    }
  //
  //    // See eq. 8.15, 8.16
  //    // Substract small number (1e-6) so that `l` is always < mipmaps - 1
  //    scalar_t l = min(lambda_, mipmaps - 1 - 1e-6);
  //
  //    // The following correction is divergence from the specification
  //    // The reason is that it is typically assumed that the full pyramid is available, but if not,
  //    // clipping of the level happens as in the line above, which causes taps to be spread with
  //    // distances higher than the size of the texel. Which in turn causes aliasing and not desirable
  //    // long-range sampling So if clipping happens, we recompute clipped Pmax and scale gradients
  //    // accordingly
  //    if (clip_grad && lambda_ > mipmaps - 1) {
  //      scalar_t p_max_corrected = exp2(l) * N;
  //      scalar_t scaling = p_max_corrected / p_max;
  //      dudx *= scaling;
  //      dvdx *= scaling;
  //      dudy *= scaling;
  //      dvdy *= scaling;
  //    }
  //
  //    l = max(l, 0.0);
  //    auto d1 = (index_t)floor(l);
  //
  //    scalar_t a = l - (scalar_t)d1;
  //
  //    index_t N_int = index_t(N);
  //    if (force_max_aniso) {
  //      N_int = max_aniso;
  //    }
  //
  //    TVec2<scalar_t> gi_acc = {scalar_t(0), scalar_t(0)};
  //
  //    if (px > py) {
  //      for (int i = 0; i < N_int; ++i) {
  //        scalar_t u_offset = dudx * ((i + 1.0) / (N_int + 1.0) * 2.0 - 1.0);
  //        scalar_t v_offset = dvdx * ((i + 1.0) / (N_int + 1.0) * 2.0 - 1.0);
  //
  //        scalar_t alpha_1 = a / N_int;
  //        scalar_t alpha_2 = (1.0 - a) / N_int;
  //
  //        if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
  //          auto ggrad = sample_bilinear_backward(
  //              inputs[d1],
  //              grad_inputs[d1],
  //              grad_output,
  //              u + u_offset,
  //              v + v_offset,
  //              w,
  //              h,
  //              n,
  //              C,
  //              alpha_2,
  //              padding_mode,
  //              align_corners,
  //              grad_input_memory_span[d1]);
  //          gi_acc += ggrad;
  //          if (mipmaps > 1) {
  //            auto ggrad2 = sample_bilinear_backward(
  //                inputs[d1 + 1],
  //                grad_inputs[d1 + 1],
  //                grad_output,
  //                u + u_offset,
  //                v + v_offset,
  //                w,
  //                h,
  //                n,
  //                C,
  //                alpha_1,
  //                padding_mode,
  //                align_corners,
  //                grad_input_memory_span[d1 + 1]);
  //            gi_acc += ggrad2;
  //          }
  //        } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
  //          auto ggrad = sample_bicubic_backward(
  //              inputs[d1],
  //              grad_inputs[d1],
  //              grad_output,
  //              u + u_offset,
  //              v + v_offset,
  //              w,
  //              h,
  //              n,
  //              C,
  //              alpha_2,
  //              padding_mode,
  //              align_corners,
  //              grad_input_memory_span[d1]);
  //          gi_acc += ggrad;
  //          if (mipmaps > 1) {
  //            auto ggrad2 = sample_bicubic_backward(
  //                inputs[d1 + 1],
  //                grad_inputs[d1 + 1],
  //                grad_output,
  //                u + u_offset,
  //                v + v_offset,
  //                w,
  //                h,
  //                n,
  //                C,
  //                alpha_1,
  //                padding_mode,
  //                align_corners,
  //                grad_input_memory_span[d1 + 1]);
  //            gi_acc += ggrad2;
  //          }
  //        }
  //      }
  //    } else {
  //      for (int i = 0; i < N_int; ++i) {
  //        scalar_t u_offset = dudy * ((i + 1.0) / (N_int + 1.0) * 2.0 - 1.0);
  //        scalar_t v_offset = dvdy * ((i + 1.0) / (N_int + 1.0) * 2.0 - 1.0);
  //
  //        scalar_t alpha_1 = a / N_int;
  //        scalar_t alpha_2 = (1.0 - a) / N_int;
  //
  //        if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
  //          auto ggrad = sample_bilinear_backward(
  //              inputs[d1],
  //              grad_inputs[d1],
  //              grad_output,
  //              u + u_offset,
  //              v + v_offset,
  //              w,
  //              h,
  //              n,
  //              C,
  //              alpha_2,
  //              padding_mode,
  //              align_corners,
  //              grad_input_memory_span[d1]);
  //          gi_acc += ggrad;
  //          if (mipmaps > 1) {
  //            auto ggrad2 = sample_bilinear_backward(
  //                inputs[d1 + 1],
  //                grad_inputs[d1 + 1],
  //                grad_output,
  //                u + u_offset,
  //                v + v_offset,
  //                w,
  //                h,
  //                n,
  //                C,
  //                alpha_1,
  //                padding_mode,
  //                align_corners,
  //                grad_input_memory_span[d1 + 1]);
  //            gi_acc += ggrad2;
  //          }
  //        } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
  //          auto ggrad = sample_bicubic_backward(
  //              inputs[d1],
  //              grad_inputs[d1],
  //              grad_output,
  //              u + u_offset,
  //              v + v_offset,
  //              w,
  //              h,
  //              n,
  //              C,
  //              alpha_2,
  //              padding_mode,
  //              align_corners,
  //              grad_input_memory_span[d1]);
  //          gi_acc += ggrad;
  //          if (mipmaps > 1) {
  //            auto ggrad2 = sample_bicubic_backward(
  //                inputs[d1 + 1],
  //                grad_inputs[d1 + 1],
  //                grad_output,
  //                u + u_offset,
  //                v + v_offset,
  //                w,
  //                h,
  //                n,
  //                C,
  //                alpha_1,
  //                padding_mode,
  //                align_corners,
  //                grad_input_memory_span[d1 + 1]);
  //            gi_acc += ggrad2;
  //          }
  //        }
  //      }
  //    }
  //
  //    // assuming grad_grid is contiguous
  //    // thus we can
  //    //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
  //    //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
  //    scalar_t* gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
  //    gGrid_ptr_NHW[0] = gi_acc.x;
  //    gGrid_ptr_NHW[1] = gi_acc.y;
  //  }
}

__host__ torch::Tensor
kernel_splatting_cuda(const torch::Tensor& input, const torch::Tensor& parameter, int64_t kernel_type) {
  TORCH_CHECK(
      input.defined() && parameter.defined(),
      "kernel_splatting(): expected input and parameter to not be undefined, but input is ",
      input,
      " and grid is ",
      parameter);
  auto input_opt = input.options();
  auto parameter_opt = parameter.options();

  TORCH_CHECK(
      input_opt.device() == parameter_opt.device() && input_opt.device().is_cuda(),
      "kernel_splatting(): expected input and parameter to be on same CUDA device, but input is on ",
      input_opt.device(),
      " and parameter is on ",
      parameter_opt.device());
  TORCH_CHECK(
      input_opt.dtype() == parameter_opt.dtype(),
      "kernel_splatting(): expected input and parameter to have same dtype, but input has ",
      input_opt.dtype(),
      " and parameter has ",
      parameter_opt.dtype());
  TORCH_CHECK(
      input_opt.layout() == torch::kStrided && parameter_opt.layout() == torch::kStrided,
      "kernel_splatting(): expected input and parameter to have torch.strided layout, but "
      "input has ",
      input_opt.layout(),
      " and parameter has ",
      parameter_opt.layout());
  TORCH_CHECK(
      input.dim() == 4 && parameter.dim() == 4,
      "kernel_splatting(): expected 4D input and parameter, but got input with sizes ",
      input.sizes(),
      " and parameter with sizes ",
      parameter.sizes());
  TORCH_CHECK(
      input.size(0) == parameter.size(0),
      "kernel_splatting(): expected input and parameter to have same batch size, "
      "but got input with sizes ",
      input.sizes(),
      " and parameter with sizes ",
      parameter.sizes());
  TORCH_CHECK(
      parameter.size(-1) == input.size(-1) && parameter.size(-2) == input.size(-2),
      "kernel_splatting(): expected input and parameter to have same width and height, ",
      "but got input with sizes ",
      input.sizes(),
      " and parameter with sizes ",
      parameter.sizes());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  auto N = input.size(0);
  auto C = input.size(1);
  auto P = parameter.size(1);
  auto H = input.size(2);
  auto W = input.size(3);
  auto output = at::zeros({N, C, H, W}, input.options());
  int64_t count = N * H * W;
  uint seed = rand();

  if (count > 0) {
    // Should be AT_DISPATCH_FLOATING_TYPES_AND_HALF, but half is broken on prod
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "kernel_splatting", [&] {
      if (at::native::canUse32BitIndexMath(input) && at::native::canUse32BitIndexMath(parameter) &&
          at::native::canUse32BitIndexMath(output)) {
        typedef int index_type;
        index_type memory_span = output.numel();

        switch (kernel_type) {
          case (int64_t)Kernel::Disk:
            kernel_splatting_kernel<scalar_t, index_type, Kernel::Disk>
                <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                    static_cast<index_type>(count),
                    getTensorInfo<scalar_t, index_type>(input),
                    getTensorInfo<scalar_t, index_type>(parameter),
                    getTensorInfo<scalar_t, index_type>(output),
                    memory_span,
                    seed);
            break;
          case (int64_t)Kernel::RadialGaussian:
            kernel_splatting_kernel<scalar_t, index_type, Kernel::RadialGaussian>
                <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                    static_cast<index_type>(count),
                    getTensorInfo<scalar_t, index_type>(input),
                    getTensorInfo<scalar_t, index_type>(parameter),
                    getTensorInfo<scalar_t, index_type>(output),
                    memory_span,
                    seed);
            break;
          case (int64_t)Kernel::Gaussian:
            kernel_splatting_kernel<scalar_t, index_type, Kernel::Gaussian>
                <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                    static_cast<index_type>(count),
                    getTensorInfo<scalar_t, index_type>(input),
                    getTensorInfo<scalar_t, index_type>(parameter),
                    getTensorInfo<scalar_t, index_type>(output),
                    memory_span,
                    seed);
            break;
          default:
            TORCH_CHECK(false, "kernel_splatting(): got unknown kernel_type: ", kernel_type);
            break;
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        typedef int64_t index_type;
        index_type memory_span = output.numel();

        switch (kernel_type) {
          case (int64_t)Kernel::Disk:
            kernel_splatting_kernel<scalar_t, index_type, Kernel::Disk>
                <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                    static_cast<index_type>(count),
                    getTensorInfo<scalar_t, index_type>(input),
                    getTensorInfo<scalar_t, index_type>(parameter),
                    getTensorInfo<scalar_t, index_type>(output),
                    memory_span,
                    seed);
            break;
          case (int64_t)Kernel::RadialGaussian:
            kernel_splatting_kernel<scalar_t, index_type, Kernel::RadialGaussian>
                <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                    static_cast<index_type>(count),
                    getTensorInfo<scalar_t, index_type>(input),
                    getTensorInfo<scalar_t, index_type>(parameter),
                    getTensorInfo<scalar_t, index_type>(output),
                    memory_span,
                    seed);
            break;
          case (int64_t)Kernel::Gaussian:
            kernel_splatting_kernel<scalar_t, index_type, Kernel::Gaussian>
                <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                    static_cast<index_type>(count),
                    getTensorInfo<scalar_t, index_type>(input),
                    getTensorInfo<scalar_t, index_type>(parameter),
                    getTensorInfo<scalar_t, index_type>(output),
                    memory_span,
                    seed);
            break;
          default:
            TORCH_CHECK(false, "kernel_splatting(): got unknown kernel_type: ", kernel_type);
            break;
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
  return output;
}

template <typename scalar_t, typename index_t, Kernel kernel_type>
__host__ void dispatch_requires_grad_kernel_splatting_backward_kernel(
    const index_t count,
    TensorInfo<scalar_t, index_t> grad_output,
    TensorInfo<scalar_t, index_t> input,
    TensorInfo<scalar_t, index_t> parameter,
    TensorInfo<scalar_t, index_t> grad_input,
    TensorInfo<scalar_t, index_t> grad_parameter,
    bool input_requires_grad,
    bool parameter_requires_grad) {
  if (input_requires_grad && parameter_requires_grad) {
    kernel_splatting_backward_kernel<scalar_t, index_t, kernel_type, true, true>
        <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            count, grad_output, input, parameter, grad_input, grad_parameter);
  } else if (!input_requires_grad && parameter_requires_grad) {
    kernel_splatting_backward_kernel<scalar_t, index_t, kernel_type, false, true>
        <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            count, grad_output, input, parameter, grad_input, grad_parameter);
  } else if (input_requires_grad && !parameter_requires_grad) {
    kernel_splatting_backward_kernel<scalar_t, index_t, kernel_type, true, false>
        <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            count, grad_output, input, parameter, grad_input, grad_parameter);
  }
}

template <typename scalar_t, typename index_t>
__host__ void dispatch_kernel_type_kernel_splatting_backward_kernel(
    const index_t count,
    TensorInfo<scalar_t, index_t> grad_output,
    TensorInfo<scalar_t, index_t> input,
    TensorInfo<scalar_t, index_t> parameter,
    TensorInfo<scalar_t, index_t> grad_input,
    TensorInfo<scalar_t, index_t> grad_parameter,
    bool input_requires_grad,
    bool parameter_requires_grad,
    Kernel kernel_type) {
  switch (kernel_type) {
    case Kernel::Disk:
      dispatch_requires_grad_kernel_splatting_backward_kernel<scalar_t, index_t, Kernel::Disk>(
          count,
          grad_output,
          input,
          parameter,
          grad_input,
          grad_parameter,
          input_requires_grad,
          parameter_requires_grad);
    case Kernel::RadialGaussian:
      dispatch_requires_grad_kernel_splatting_backward_kernel<scalar_t, index_t, Kernel::RadialGaussian>(
          count,
          grad_output,
          input,
          parameter,
          grad_input,
          grad_parameter,
          input_requires_grad,
          parameter_requires_grad);
    case Kernel::Gaussian:
      dispatch_requires_grad_kernel_splatting_backward_kernel<scalar_t, index_t, Kernel::Gaussian>(
          count,
          grad_output,
          input,
          parameter,
          grad_input,
          grad_parameter,
          input_requires_grad,
          parameter_requires_grad);
  }
}

__host__ std::tuple<torch::Tensor, torch::Tensor> kernel_splatting_cuda_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& parameter,
    int64_t kernel_type,
    bool input_requires_grad,
    bool parameter_requires_grad) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  auto N = input.size(0);
  auto C = input.size(1);
  auto P = parameter.size(1);
  auto H = input.size(2);
  auto W = input.size(3);
  auto output = at::zeros({N, C, H, W}, input.options());
  int64_t count = N * H * W;

  torch::Tensor grad_input = torch::Tensor();
  torch::Tensor grad_parameter = torch::Tensor();

  if (!grad_output.defined()) {
    return std::make_tuple(torch::Tensor(), torch::Tensor());
  }

  if (input_requires_grad || true)
    grad_input = at::zeros_like(input, at::MemoryFormat::Contiguous);

  if (parameter_requires_grad || true)
    grad_parameter = at::zeros_like(parameter, at::MemoryFormat::Contiguous);

  if (count > 0) {
    // Should be AT_DISPATCH_FLOATING_TYPES_AND_HALF, but half is broken on prod
    AT_DISPATCH_FLOATING_TYPES(input[0].scalar_type(), "kernel_splatting_cuda_backward", [&] {
      if (at::native::canUse32BitIndexMath(input) && at::native::canUse32BitIndexMath(parameter) &&
          at::native::canUse32BitIndexMath(grad_output)) {
        typedef int index_type;

        dispatch_kernel_type_kernel_splatting_backward_kernel<scalar_t, index_type>(
            static_cast<index_type>(count),
            getTensorInfo<scalar_t, index_type>(grad_output),
            getTensorInfo<scalar_t, index_type>(input),
            getTensorInfo<scalar_t, index_type>(parameter),
            getTensorInfo<scalar_t, index_type>(grad_input),
            getTensorInfo<scalar_t, index_type>(grad_parameter),
            input_requires_grad,
            parameter_requires_grad,
            (Kernel)kernel_type);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        typedef int64_t index_type;

        dispatch_kernel_type_kernel_splatting_backward_kernel<scalar_t, index_type>(
            static_cast<index_type>(count),
            getTensorInfo<scalar_t, index_type>(grad_output),
            getTensorInfo<scalar_t, index_type>(input),
            getTensorInfo<scalar_t, index_type>(parameter),
            getTensorInfo<scalar_t, index_type>(grad_input),
            getTensorInfo<scalar_t, index_type>(grad_parameter),
            input_requires_grad,
            parameter_requires_grad,
            (Kernel)kernel_type);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
  return std::make_tuple(grad_input, grad_parameter);
}
