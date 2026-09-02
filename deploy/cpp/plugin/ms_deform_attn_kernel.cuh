/*!
 * Forward-only multi-scale deformable attention CUDA kernel (no ATen).
 * Adapted from Deformable-DETR / SenseTime ms_deform_im2col (Apache-2.0).
 */
#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>

namespace det_seg_kp {
namespace msda {

#define MSDA_CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;      \
       i < (n);                                            \
       i += blockDim.x * gridDim.x)

constexpr int kCudaNumThreads = 1024;

inline int get_blocks(int n, int num_threads = kCudaNumThreads) {
  return (n + num_threads - 1) / num_threads;
}

template <typename T>
__device__ inline float to_f(T v);

template <>
__device__ inline float to_f<float>(float v) {
  return v;
}

template <>
__device__ inline float to_f<__half>(__half v) {
  return __half2float(v);
}

template <typename T>
__device__ inline T from_f(float v);

template <>
__device__ inline float from_f<float>(float v) {
  return v;
}

template <>
__device__ inline __half from_f<__half>(float v) {
  return __float2half(v);
}

template <typename T>
__device__ T ms_deform_attn_im2col_bilinear(
    const T* bottom_data,
    const int height,
    const int width,
    const int nheads,
    const int channels,
    const T h,
    const T w,
    const int m,
    const int c) {
  const float hf = to_f(h);
  const float wf = to_f(w);
  const int h_low = static_cast<int>(floorf(hf));
  const int w_low = static_cast<int>(floorf(wf));
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = hf - static_cast<float>(h_low);
  const float lw = wf - static_cast<float>(w_low);
  const float hh = 1.f - lh;
  const float hw = 1.f - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  float v1 = 0.f;
  if (h_low >= 0 && w_low >= 0) {
    v1 = to_f(bottom_data[h_low_ptr_offset + w_low_ptr_offset + base_ptr]);
  }
  float v2 = 0.f;
  if (h_low >= 0 && w_high <= width - 1) {
    v2 = to_f(bottom_data[h_low_ptr_offset + w_high_ptr_offset + base_ptr]);
  }
  float v3 = 0.f;
  if (h_high <= height - 1 && w_low >= 0) {
    v3 = to_f(bottom_data[h_high_ptr_offset + w_low_ptr_offset + base_ptr]);
  }
  float v4 = 0.f;
  if (h_high <= height - 1 && w_high <= width - 1) {
    v4 = to_f(bottom_data[h_high_ptr_offset + w_high_ptr_offset + base_ptr]);
  }

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  return from_f<T>(w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}

template <typename T>
__global__ void ms_deformable_im2col_gpu_kernel(
    const int n,
    const T* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const T* data_sampling_loc,
    const T* data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    T* data_col) {
  (void)batch_size;
  (void)spatial_size;
  MSDA_CUDA_KERNEL_LOOP(index, n) {
    int temp = index;
    const int c_col = temp % channels;
    temp /= channels;
    const int sampling_index = temp;
    const int m_col = temp % num_heads;
    temp /= num_heads;
    const int q_col = temp % num_query;
    temp /= num_query;
    const int b_col = temp;
    (void)q_col;

    T* data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    T col = static_cast<T>(0);

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = static_cast<int>(data_level_start_index[l_col]);
      const int spatial_h = static_cast<int>(data_spatial_shapes[l_col * 2]);
      const int spatial_w = static_cast<int>(data_spatial_shapes[l_col * 2 + 1]);
      const T* data_value_ptr =
          data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col = 0; p_col < num_point; ++p_col) {
        const T loc_w = data_sampling_loc[data_loc_w_ptr];
        const T loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const T weight = data_attn_weight[data_weight_ptr];

        const float h_im =
            to_f(loc_h) * static_cast<float>(spatial_h) - 0.5f;
        const float w_im =
            to_f(loc_w) * static_cast<float>(spatial_w) - 0.5f;

        if (h_im > -1.f && w_im > -1.f && h_im < static_cast<float>(spatial_h) &&
            w_im < static_cast<float>(spatial_w)) {
          col = from_f<T>(
              to_f(col) +
              to_f(ms_deform_attn_im2col_bilinear(
                  data_value_ptr, spatial_h, spatial_w, num_heads, channels,
                  from_f<T>(h_im), from_f<T>(w_im), m_col, c_col)) *
                  to_f(weight));
        }
        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}

template <typename T>
void ms_deformable_im2col_cuda(
    cudaStream_t stream,
    const T* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const T* data_sampling_loc,
    const T* data_attn_weight,
    int batch_size,
    int spatial_size,
    int num_heads,
    int channels,
    int num_levels,
    int num_query,
    int num_point,
    T* data_col) {
  const int num_kernels = batch_size * num_query * num_heads * channels;
  ms_deformable_im2col_gpu_kernel<T>
      <<<get_blocks(num_kernels), kCudaNumThreads, 0, stream>>>(
          num_kernels, data_value, data_spatial_shapes, data_level_start_index,
          data_sampling_loc, data_attn_weight, batch_size, spatial_size, num_heads,
          channels, num_levels, num_query, num_point, data_col);
}

/** Host entry: value [B,S,M,D] → output [B,Q,M*D], same layout as training CUDA. */
template <typename T>
void ms_deform_attn_cuda_forward(
    cudaStream_t stream,
    const T* value,
    const int64_t* spatial_shapes,
    const int64_t* level_start_index,
    const T* sampling_loc,
    const T* attn_weight,
    T* output,
    int batch,
    int spatial_size,
    int num_heads,
    int channels,
    int num_levels,
    int num_query,
    int num_point,
    int im2col_step) {
  const int im2col_step_ = std::min(batch, im2col_step);
  if (batch % im2col_step_ != 0) {
    std::fprintf(stderr, "ms_deform_attn: batch (%d) must divide im2col_step (%d)\n",
                 batch, im2col_step_);
    return;
  }

  // output viewed as [batch/im2col_step_, im2col_step_, Q, M, D] during kernel;
  // we write [B, Q, M, D] then caller treats as [B, Q, M*D].
  const int batch_n = im2col_step_;
  const size_t per_value_size =
      static_cast<size_t>(spatial_size) * num_heads * channels;
  const size_t per_sample_loc_size = static_cast<size_t>(num_query) * num_heads *
                                     num_levels * num_point * 2;
  const size_t per_attn_weight_size =
      static_cast<size_t>(num_query) * num_heads * num_levels * num_point;
  const size_t per_out_size =
      static_cast<size_t>(num_query) * num_heads * channels;

  for (int n = 0; n < batch / im2col_step_; ++n) {
    ms_deformable_im2col_cuda<T>(
        stream,
        value + n * im2col_step_ * per_value_size,
        spatial_shapes,
        level_start_index,
        sampling_loc + n * im2col_step_ * per_sample_loc_size,
        attn_weight + n * im2col_step_ * per_attn_weight_size,
        batch_n,
        spatial_size,
        num_heads,
        channels,
        num_levels,
        num_query,
        num_point,
        output + n * im2col_step_ * per_out_size);
  }
}

}  // namespace msda
}  // namespace det_seg_kp
