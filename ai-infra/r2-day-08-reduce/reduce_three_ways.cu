// CUDA/H100 execution not validated in the authoring environment.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

constexpr int kWarpSize = 32;

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t error__ = (call);                                                \
    if (error__ != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(error__) << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                           \
  } while (0)

__global__ void reduce_atomic(const float *input, float *output, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    atomicAdd(output, input[index]);
  }
}

__global__ void reduce_shared_tree(const float *input, float *output, int n) {
  extern __shared__ float tile[];
  int tid = threadIdx.x;
  int index = blockIdx.x * blockDim.x + tid;
  tile[tid] = index < n ? input[index] : 0.0f;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      tile[tid] += tile[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(output, tile[0]);
  }
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ void reduce_warp_shuffle(const float *input, float *output, int n) {
  extern __shared__ float warp_sums[];
  int tid = threadIdx.x;
  int lane = tid & (kWarpSize - 1);
  int warp = tid / kWarpSize;
  int warp_count = blockDim.x / kWarpSize;
  int index = blockIdx.x * blockDim.x + tid;

  float value = index < n ? input[index] : 0.0f;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = lane < warp_count ? warp_sums[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      atomicAdd(output, block_sum);
    }
  }
}

enum class Variant { kAtomic, kSharedTree, kWarpShuffle };

const char *variant_name(Variant variant) {
  switch (variant) {
    case Variant::kAtomic:
      return "atomic-per-element";
    case Variant::kSharedTree:
      return "shared-tree";
    case Variant::kWarpShuffle:
      return "warp-shuffle";
  }
  return "unknown";
}

void launch(Variant variant, const float *input, float *output, int n,
            int block_size) {
  int grid_size = (n + block_size - 1) / block_size;
  if (variant == Variant::kAtomic) {
    reduce_atomic<<<grid_size, block_size>>>(input, output, n);
  } else if (variant == Variant::kSharedTree) {
    size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);
    reduce_shared_tree<<<grid_size, block_size, shared_bytes>>>(input, output, n);
  } else {
    size_t shared_bytes =
        static_cast<size_t>(block_size / kWarpSize) * sizeof(float);
    reduce_warp_shuffle<<<grid_size, block_size, shared_bytes>>>(input, output,
                                                                 n);
  }
  CUDA_CHECK(cudaGetLastError());
}

float benchmark(Variant variant, const float *input, float *output, int n,
                int block_size, int warmups, int repeats) {
  for (int i = 0; i < warmups; ++i) {
    CUDA_CHECK(cudaMemset(output, 0, sizeof(float)));
    launch(variant, input, output, n, block_size);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> milliseconds;
  milliseconds.reserve(repeats);

  for (int i = 0; i < repeats; ++i) {
    CUDA_CHECK(cudaMemset(output, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    launch(variant, input, output, n, block_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    milliseconds.push_back(elapsed);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::sort(milliseconds.begin(), milliseconds.end());
  return milliseconds[milliseconds.size() / 2];
}

int main(int argc, char **argv) {
  int n = argc > 1 ? std::atoi(argv[1]) : (1 << 22);
  int block_size = argc > 2 ? std::atoi(argv[2]) : 256;
  if (n <= 0 || block_size < 32 || block_size > 1024 ||
      (block_size & (block_size - 1)) != 0 || block_size % 32 != 0) {
    std::cerr << "usage: " << argv[0]
              << " [positive_elements] [power_of_two_block_multiple_of_32]"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<float> host_input(n, 1.0f);
  float *device_input = nullptr;
  float *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_input),
                        host_input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_output),
                        sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, host_input.data(),
                        host_input.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  const int warmups = 5;
  const int repeats = 21;
  for (Variant variant : {Variant::kAtomic, Variant::kSharedTree,
                          Variant::kWarpShuffle}) {
    float median_ms = benchmark(variant, device_input, device_output, n,
                                block_size, warmups, repeats);
    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, device_output, sizeof(float),
                          cudaMemcpyDeviceToHost));
    float expected = static_cast<float>(n);
    bool correct = std::fabs(result - expected) <= 0.5f;
    std::cout << variant_name(variant) << " result=" << result
              << " expected=" << expected << " correct=" << std::boolalpha
              << correct << " median_kernel_ms=" << median_ms << std::endl;
    if (!correct) {
      CUDA_CHECK(cudaFree(device_input));
      CUDA_CHECK(cudaFree(device_output));
      return EXIT_FAILURE;
    }
  }

  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  return EXIT_SUCCESS;
}
