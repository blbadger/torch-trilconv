#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace extension_cpp {


__global__ void trilconv_kernel(int numel, const float* input, const float* conv_weights, float* result){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    for (int i=0; i<idx; i++){
      result[idx] += input[i] * conv_weights[i];
    }
  }
}

at::Tensor trilconv_kernel(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  int numel = a_contig.numel();
  mul_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, result_ptr);
  return result;
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("trilconv", &trilconv_kernel);
}
}
