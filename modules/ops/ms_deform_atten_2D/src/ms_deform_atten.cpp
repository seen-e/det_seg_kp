/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include <torch/extension.h>
#include <vector>

#include "ms_deform_atten_cuda.h"

at::Tensor ms_deform_attn_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
{
    if (value.is_cuda())
    {
        return ms_deform_attn_cuda_forward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor> ms_deform_attn_backward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step)
{
    if (value.is_cuda())
    {
        return ms_deform_attn_cuda_backward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
    }
    AT_ERROR("Not implemented on the CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");
}
