#include "ms_deform_attn_kernel.cuh"

#include <cuda_fp16.h>

namespace det_seg_kp {
namespace msda {

template void ms_deform_attn_cuda_forward<float>(
    cudaStream_t, const float*, const int64_t*, const int64_t*, const float*,
    const float*, float*, int, int, int, int, int, int, int, int);

template void ms_deform_attn_cuda_forward<__half>(
    cudaStream_t, const __half*, const int64_t*, const int64_t*, const __half*,
    const __half*, __half*, int, int, int, int, int, int, int, int);

}  // namespace msda
}  // namespace det_seg_kp
