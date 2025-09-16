/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <optional>

namespace impl {
namespace generic {
namespace native {
namespace {
using ::executorch::runtime::getLeadingDims;

#define ET_FORALL_CADENCE_QUANTIZED_TYPES(_) \
  _(uint8_t, Byte)                           \
  _(int8_t, Char)

inline __attribute__((always_inline)) void linear_(
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& weight,
    const std::optional<::executorch::aten::Tensor>& bias,
    ::executorch::aten::Tensor& output) {
  const float* __restrict__ input_data = input.const_data_ptr<float>();
  const float* __restrict__ weight_data = weight.const_data_ptr<float>();
  const float* __restrict__ bias_data = bias.value().const_data_ptr<float>();
  float* __restrict__ output_data = output.mutable_data_ptr<float>();

  // input comes in shape [batch_size, in_dim]
  // weight comes in shape [out_dim, in_dim]
  // output comes in empty with shape [batch_size, out_dim]
  // Perform matrix multiply (M x N) x (N x P) => M x P
  int64_t M = weight.size(0); // = out_dim
  int64_t N = weight.size(1); // = in_dim

  // Given an N-dimensional input [d0, d1, d2, ..., d_{N-2}, d_{N-1}], the
  // leading dimensions is d0 * d1 * ... * d_{N-2}
  int64_t leading_dims = getLeadingDims(input, input.dim() - 1);

  for (int i = 0; i < leading_dims; ++i) {
    for (int j = 0; j < M; ++j) {
      float sum = bias_data[j];
      for (int k = 0; k < N; ++k) {
        sum += input_data[i * N + k] * weight_data[j * N + k];
      }
      output_data[i * M + j] = sum;
    }
  }
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl

// Forward declarations for quantized operators
namespace impl {
namespace generic {
namespace native {

using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

// Quantized ReLU operators
void quantized_relu_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& in_zero_point,
    const int64_t out_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    Tensor& output);

void quantized_relu_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output);

void quantized_relu_asym8s_asym8s_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output);

void quantized_relu_asym8u_asym8u_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output);

// Quantized Add operators
void quantized_add_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    const double X_scale,
    const int64_t X_zero_point,
    const Tensor& Y,
    const double Y_scale,
    const int64_t Y_zero_point,
    const double out_scale,
    const int64_t out_zero_point,
    Tensor& output);

void quantized_add_asym8sxasym8s_asym8s_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    const double X_scale,
    const int64_t X_zero_point,
    const Tensor& Y,
    const double Y_scale,
    const int64_t Y_zero_point,
    const double out_scale,
    const int64_t out_zero_point,
    Tensor& output);

void quantized_add_asym8uxasym8u_asym8u_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    const double X_scale,
    const int64_t X_zero_point,
    const Tensor& Y,
    const double Y_scale,
    const int64_t Y_zero_point,
    const double out_scale,
    const int64_t out_zero_point,
    Tensor& output);

// Quantized MatMul operators
void quantized_matmul_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    const int64_t X_zero_point,
    const Tensor& Y,
    const int64_t Y_zero_point,
    const std::optional<Tensor>& bias,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    const bool transposed,
    Tensor& output);

void quantized_matmul_asym8sxasym8s_asym8s_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    const int64_t X_zero_point,
    const Tensor& Y,
    const int64_t Y_zero_point,
    const std::optional<Tensor>& bias,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    const bool transposed,
    Tensor& output);

void quantized_matmul_asym8uxasym8u_asym8u_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    const int64_t X_zero_point,
    const Tensor& Y,
    const int64_t Y_zero_point,
    const std::optional<Tensor>& bias,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    const bool transposed,
    Tensor& output);

} // namespace native
} // namespace generic
} // namespace impl
// Forward declarations for quantized operators
namespace impl {
namespace generic {
namespace native {

using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

// Quantized ReLU operators
void quantized_relu_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& in_zero_point,
    const int64_t out_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    Tensor& output);

void quantized_relu_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output);

void quantized_relu_asym8s_asym8s_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output);

void quantized_relu_asym8u_asym8u_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output);

// Quantized Add operators
void quantized_add_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    const double X_scale,
    const int64_t X_zero_point,
    const Tensor& Y,
    const double Y_scale,
    const int64_t Y_zero_point,
    const double out_scale,
    const int64_t out_zero_point,
    Tensor& output);

void quantized_add_asym8sxasym8s_asym8s_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    const double X_scale,
    const int64_t X_zero_point,
    const Tensor& Y,
    const double Y_scale,
    const int64_t Y_zero_point,
    const double out_scale,
    const int64_t out_zero_point,
    Tensor& output);

void quantized_add_asym8uxasym8u_asym8u_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    const double X_scale,
    const int64_t X_zero_point,
    const Tensor& Y,
    const double Y_scale,
    const int64_t Y_zero_point,
    const double out_scale,
    const int64_t out_zero_point,
    Tensor& output);

// Quantized MatMul operators
void quantized_matmul_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    const int64_t X_zero_point,
    const Tensor& Y,
    const int64_t Y_zero_point,
    const std::optional<Tensor>& bias,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    const bool transposed,
    Tensor& output);

void quantized_matmul_asym8sxasym8s_asym8s_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    const int64_t X_zero_point,
    const Tensor& Y,
    const int64_t Y_zero_point,
    const std::optional<Tensor>& bias,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    const bool transposed,
    Tensor& output);

void quantized_matmul_asym8uxasym8u_asym8u_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    const int64_t X_zero_point,
    const Tensor& Y,
    const int64_t Y_zero_point,
    const std::optional<Tensor>& bias,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    const bool transposed,
    Tensor& output);

} // namespace native
} // namespace generic
} // namespace impl

// Forward declarations for quantized operators
namespace impl {
namespace generic {
namespace native {

using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

// Quantized ReLU operators
void quantized_relu_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& in_zero_point,
    const int64_t out_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    Tensor& output);

void quantized_relu_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output);

void quantized_relu_asym8s_asym8s_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output);

void quantized_relu_asym8u_asym8u_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output);

// Quantized Add operators
void quantized_add_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    const double X_scale,
    const int64_t X_zero_point,
    const Tensor& Y,
    const double Y_scale,
    const int64_t Y_zero_point,
    const double out_scale,
    const int64_t out_zero_point,
    Tensor& output);

void quantized_add_asym8sxasym8s_asym8s_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    const double X_scale,
    const int64_t X_zero_point,
    const Tensor& Y,
    const double Y_scale,
    const int64_t Y_zero_point,
    const double out_scale,
    const int64_t out_zero_point,
    Tensor& output);

void quantized_add_asym8uxasym8u_asym8u_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    const double X_scale,
    const int64_t X_zero_point,
    const Tensor& Y,
    const double Y_scale,
    const int64_t Y_zero_point,
    const double out_scale,
    const int64_t out_zero_point,
    Tensor& output);

// Quantized MatMul operators
void quantized_matmul_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    const int64_t X_zero_point,
    const Tensor& Y,
    const int64_t Y_zero_point,
    const std::optional<Tensor>& bias,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    const bool transposed,
    Tensor& output);

void quantized_matmul_asym8sxasym8s_asym8s_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    const int64_t X_zero_point,
    const Tensor& Y,
    const int64_t Y_zero_point,
    const std::optional<Tensor>& bias,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    const bool transposed,
    Tensor& output);

void quantized_matmul_asym8uxasym8u_asym8u_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    const int64_t X_zero_point,
    const Tensor& Y,
    const int64_t Y_zero_point,
    const std::optional<Tensor>& bias,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    const bool transposed,
    Tensor& output);

} // namespace native
} // namespace generic
} // namespace impl
