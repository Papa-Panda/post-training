// CUDA/H100 execution not validated in the authoring environment.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t error = (call);                                                  \
    if (error != cudaSuccess) {                                                  \
      std::cerr << #call << " failed: " << cudaGetErrorString(error) << '\n';   \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                           \
  } while (0)

__global__ void vector_add(const float* a, const float* b, float* c,
                           size_t logical_elements, int stride) {
  const size_t logical_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (logical_index < logical_elements) {
    const size_t physical_index = logical_index * static_cast<size_t>(stride);
    c[physical_index] = a[physical_index] + b[physical_index];
  }
}

float run_case(size_t logical_elements, int stride, int threads_per_block) {
  const size_t storage_elements = logical_elements * static_cast<size_t>(stride);
  const size_t bytes = storage_elements * sizeof(float);
  std::vector<float> host_a(storage_elements, 1.0f);
  std::vector<float> host_b(storage_elements, 2.0f);
  std::vector<float> host_c(storage_elements, 0.0f);

  float *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_a), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_b), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_c), bytes));
  CUDA_CHECK(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(device_c, 0, bytes));

  const int blocks = static_cast<int>(
      (logical_elements + threads_per_block - 1) / threads_per_block);
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  vector_add<<<blocks, threads_per_block>>>(device_a, device_b, device_c,
                                            logical_elements, stride);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));

  for (size_t logical_index = 0; logical_index < logical_elements;
       ++logical_index) {
    const size_t physical_index = logical_index * static_cast<size_t>(stride);
    if (std::fabs(host_c[physical_index] - 3.0f) > 1e-6f) {
      std::cerr << "validation failed at physical index " << physical_index << '\n';
      std::exit(EXIT_FAILURE);
    }
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_c));
  return elapsed_ms;
}

int main() {
  constexpr size_t kLogicalElements = 1 << 20;
  for (int threads_per_block : {128, 256, 512}) {
    for (int stride : {1, 2}) {
      const float elapsed_ms =
          run_case(kLogicalElements, stride, threads_per_block);
      std::cout << "block_size=" << threads_per_block << " stride=" << stride
                << " validated=" << kLogicalElements
                << " elements elapsed_ms=" << elapsed_ms << '\n';
    }
  }
  return 0;
}
