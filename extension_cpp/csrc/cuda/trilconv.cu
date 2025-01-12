#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace extension_cpp {


__global__ void trilconv_kernel(int numel, const float* input, const float* conv_weights, float* result){
  int m = blockIdx.x;
  int h = (blockIdx.y / W_grid)*TILE_WIDTH + threadIdx.y;
  int w = (blockIdx.y % W_grid)*TILE_WIDTH + threadIdx.x;
  int n = blockIdx.z;

  float acc = 0.;
  for (int c = 0; c < C; c++) {
  	for (int p = 0; p < K; p++){
      for (int q = 0; q < K; q++){
      	acc += X[n, c, h+p, w+q] * W[m, c, p, q]
      }
  	}
  }
  Y[n, m, h, w] = acc;
}

at::Tensor trilconv_cuda(const at::Tensor& input, const at::Tensor& weight) {
  TORCH_CHECK(input.sizes() == weight.sizes());
  TORCH_CHECK(input.dtype() == at::kFloat);
  TORCH_CHECK(weight.dtype() == at::kFloat);  
  TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(weight.device().type() == at::DeviceType::CUDA);
  at::Tensor input_contig = input.contiguous();
  at::Tensor weight_contig = weight.contiguous();
  at::Tensor result = torch::empty(input_contig.sizes(), input_contig.options());
  const float* input_ptr = input_contig.data_ptr<float>();
  const float* weight_ptr = weight_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  int numel = input_contig.numel();
  trilconv_kernel<<<(numel+255)/256, 256>>>(numel, input_ptr, weight_ptr, result_ptr);
  return result;
}

// Registers CUDA implementations for trilconv
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("trilconv", &trilconv_cuda);
}
}
