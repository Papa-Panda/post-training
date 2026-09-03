// CUDA/H100 execution not validated in the authoring environment.
// Row-major SGEMM: C[M,N] = A[M,K] * B[K,N].
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

constexpr int kTile = 16;

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t error__ = (call);                                                \
    if (error__ != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(error__) << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                           \
  } while (0)

#define CUBLAS_CHECK(call)                                                       \
  do {                                                                          \
    cublasStatus_t status__ = (call);                                            \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                                     \
      std::cerr << "cuBLAS error code " << static_cast<int>(status__) << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                           \
  } while (0)

__global__ void gemm_naive(const float* a, const float* b, float* c,
                           int m, int n, int k) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;

  float acc = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    acc += a[row * k + inner] * b[inner * n + col];
  }
  c[row * n + col] = acc;
}

__global__ void gemm_tiled(const float* a, const float* b, float* c,
                           int m, int n, int k) {
  __shared__ float a_tile[kTile][kTile];
  __shared__ float b_tile[kTile][kTile];

  const int local_row = threadIdx.y;
  const int local_col = threadIdx.x;
  const int row = blockIdx.y * kTile + local_row;
  const int col = blockIdx.x * kTile + local_col;
  float acc = 0.0f;

  for (int inner0 = 0; inner0 < k; inner0 += kTile) {
    const int a_col = inner0 + local_col;
    const int b_row = inner0 + local_row;
    a_tile[local_row][local_col] =
        (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    b_tile[local_row][local_col] =
        (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int inner = 0; inner < kTile; ++inner) {
      acc += a_tile[local_row][inner] * b_tile[inner][local_col];
    }
    __syncthreads();
  }

  if (row < m && col < n) c[row * n + col] = acc;
}

enum class Variant { kNaive, kTiled, kCublas };

const char* variant_name(Variant variant) {
  switch (variant) {
    case Variant::kNaive: return "naive";
    case Variant::kTiled: return "shared-tiled";
    case Variant::kCublas: return "cublas-sgemm";
  }
  return "unknown";
}

void launch(Variant variant, cublasHandle_t handle, const float* a,
            const float* b, float* c, int m, int n, int k) {
  if (variant == Variant::kNaive || variant == Variant::kTiled) {
    const dim3 block(kTile, kTile);
    const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
    if (variant == Variant::kNaive) {
      gemm_naive<<<grid, block>>>(a, b, c, m, n, k);
    } else {
      gemm_tiled<<<grid, block>>>(a, b, c, m, n, k);
    }
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  // cuBLAS is column-major. Swapping A/B computes row-major C through
  // C^T[N,M] = B^T[N,K] * A^T[K,M] without transposing storage.
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           n, m, k, &alpha, b, n, a, k, &beta, c, n));
}

float benchmark(Variant variant, cublasHandle_t handle, const float* a,
                const float* b, float* c, int m, int n, int k,
                int warmups, int repeats) {
  for (int i = 0; i < warmups; ++i) launch(variant, handle, a, b, c, m, n, k);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> milliseconds;
  milliseconds.reserve(repeats);
  for (int i = 0; i < repeats; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch(variant, handle, a, b, c, m, n, k);
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

float max_abs_error(const std::vector<float>& got,
                    const std::vector<float>& reference) {
  float maximum = 0.0f;
  for (size_t i = 0; i < got.size(); ++i) {
    maximum = std::max(maximum, std::fabs(got[i] - reference[i]));
  }
  return maximum;
}

int main(int argc, char** argv) {
  const int m = argc > 1 ? std::atoi(argv[1]) : 512;
  const int n = argc > 2 ? std::atoi(argv[2]) : m;
  const int k = argc > 3 ? std::atoi(argv[3]) : m;
  if (m <= 0 || n <= 0 || k <= 0) {
    std::cerr << "usage: " << argv[0] << " [positive_M] [positive_N] [positive_K]\n";
    return EXIT_FAILURE;
  }

  std::vector<float> host_a(static_cast<size_t>(m) * k);
  std::vector<float> host_b(static_cast<size_t>(k) * n);
  for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = static_cast<float>(static_cast<int>(i % 7) - 3) / 8.0f;
  for (size_t i = 0; i < host_b.size(); ++i) host_b[i] = static_cast<float>(static_cast<int>(i % 5) - 2) / 8.0f;

  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_a), host_a.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_b), host_b.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_c), static_cast<size_t>(m) * n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_a, host_a.data(), host_a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b, host_b.data(), host_b.size() * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  std::cout << "cuBLAS_math_mode=CUBLAS_DEFAULT_MATH" << std::endl;
  const int warmups = 5;
  const int repeats = 21;
  std::vector<float> reference(static_cast<size_t>(m) * n);

  for (Variant variant : {Variant::kCublas, Variant::kNaive, Variant::kTiled}) {
    const float median_ms = benchmark(variant, handle, device_a, device_b,
                                      device_c, m, n, k, warmups, repeats);
    std::vector<float> output(static_cast<size_t>(m) * n);
    CUDA_CHECK(cudaMemcpy(output.data(), device_c, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (variant == Variant::kCublas) reference = output;
    const float error = max_abs_error(output, reference);
    const double tflops = (2.0 * m * n * k) / (median_ms * 1.0e9);
    std::cout << variant_name(variant) << " M=" << m << " N=" << n << " K=" << k
              << " median_kernel_ms=" << median_ms << " TFLOP/s=" << tflops
              << " max_abs_error_vs_cublas=" << error << std::endl;
    if (variant != Variant::kCublas && error > 1.0e-3f * std::max(1, k)) {
      std::cerr << "correctness check failed\n";
      return EXIT_FAILURE;
    }
  }

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_c));
  return EXIT_SUCCESS;
}
