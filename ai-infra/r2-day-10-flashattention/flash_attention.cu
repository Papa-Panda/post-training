// CUDA/H100 execution not validated in the authoring environment.
//
// WHAT THIS FILE IS: the BASELINE that FlashAttention replaces -- a tiled
// two-pass attention that materializes the full N x N score matrix S in HBM:
//   pass 1: S = scale * Q @ K^T   (shared-memory tiled GEMM, same pattern as
//           r2-day-09 tiled GEMM, transposed-B variant)
//   pass 2: P = rowwise softmax(S)
//   pass 3: O = P @ V            (shared-memory tiled GEMM)
// Its HBM payload is exactly the 4N^2 + 4Nd (fp32 elements) model counted in
// the lesson README. The fused single-pass online-softmax kernel is the
// lesson's whiteboard subject and is modeled by the Python loop
// (flash_models.py), which mirrors the paper's Algorithm 1 loop structure.
// A fused CUDA kernel is NOT shipped here because it cannot be compiled or
// run in this environment, and shipping an unvalidated kernel would violate
// the repo's quality bar.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t error__ = (call);                                                \
    if (error__ != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(error__) << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                          \
  } while (0)

constexpr int kTile = 16;

// S[row][col] = scale * sum_d Q[row][d] * K[col][d].  Q,K: N x D row-major.
__global__ void attn_scores_tiled(const float *Q, const float *K, float *S,
                                  int N, int D, float scale) {
  __shared__ float Qs[kTile][kTile + 1];  // +1 pad avoids bank conflicts
  __shared__ float Ks[kTile][kTile + 1];
  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < (D + kTile - 1) / kTile; ++t) {
    const int qd = t * kTile + threadIdx.x;
    const int kd = t * kTile + threadIdx.y;
    Qs[threadIdx.y][threadIdx.x] =
        (row < N && qd < D) ? Q[row * D + qd] : 0.0f;
    Ks[threadIdx.y][threadIdx.x] =
        (col < N && kd < D) ? K[col * D + kd] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kTile; ++k) {
      acc += Qs[threadIdx.y][k] * Ks[k][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < N && col < N) {
    S[row * N + col] = acc * scale;
  }
}

// One thread per row (demo sizes are small; production uses one block per row
// with an online reduction -- the same running-max idea as online softmax).
__global__ void row_softmax_stable(const float *S, float *P, int N) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N) return;
  float m = -1e30f;
  for (int j = 0; j < N; ++j) m = fmaxf(m, S[row * N + j]);
  float sum = 0.0f;
  for (int j = 0; j < N; ++j) {
    const float e = expf(S[row * N + j] - m);
    P[row * N + j] = e;
    sum += e;
  }
  for (int j = 0; j < N; ++j) P[row * N + j] /= sum;
}

// O[row][col] = sum_j P[row][j] * V[j][col].  P: N x N, V: N x D.
__global__ void attn_apply_tiled(const float *P, const float *V, float *O,
                                 int N, int D) {
  __shared__ float Ps[kTile][kTile + 1];
  __shared__ float Vs[kTile][kTile + 1];
  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < (N + kTile - 1) / kTile; ++t) {
    const int pj = t * kTile + threadIdx.x;
    const int vj = t * kTile + threadIdx.y;
    Ps[threadIdx.y][threadIdx.x] =
        (row < N && pj < N) ? P[row * N + pj] : 0.0f;
    Vs[threadIdx.y][threadIdx.x] =
        (vj < N && col < D) ? V[vj * D + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kTile; ++k) {
      acc += Ps[threadIdx.y][k] * Vs[k][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < N && col < D) {
    O[row * D + col] = acc;
  }
}

// Host-side reference: the exact math the kernels must reproduce.
static void cpu_reference(const std::vector<float> &Q,
                          const std::vector<float> &K,
                          const std::vector<float> &V, std::vector<float> &O,
                          int N, int D, float scale) {
  std::vector<float> S(N * N), P(N * N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int d = 0; d < D; ++d) acc += Q[i * D + d] * K[j * D + d];
      S[i * N + j] = acc * scale;
    }
  for (int i = 0; i < N; ++i) {
    float m = -1e30f;
    for (int j = 0; j < N; ++j) m = std::max(m, S[i * N + j]);
    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
      const float e = std::exp(S[i * N + j] - m);
      P[i * N + j] = e;
      sum += e;
    }
    for (int j = 0; j < N; ++j) P[i * N + j] /= sum;
  }
  for (int i = 0; i < N; ++i)
    for (int c = 0; c < D; ++c) {
      float acc = 0.0f;
      for (int j = 0; j < N; ++j) acc += P[i * N + j] * V[j * D + c];
      O[i * D + c] = acc;
    }
}

int main(int argc, char **argv) {
  const int N = argc > 1 ? std::atoi(argv[1]) : 8;
  const int D = argc > 2 ? std::atoi(argv[2]) : 4;
  const float scale = 1.0f;  // kept 1 to match the lesson's hand example
  if (N <= 0 || D <= 0) {
    std::cerr << "usage: flash_attention N D\n";
    return EXIT_FAILURE;
  }
  std::vector<float> Q(N * D), K(N * D), V(N * D), O(N * D), Oref(N * D);
  for (int i = 0; i < N; ++i)
    for (int d = 0; d < D; ++d) {
      Q[i * D + d] = static_cast<float>((i * 7 + d * 3) % 5);
      K[i * D + d] = static_cast<float>((i * 5 - d * 2 + 25) % 5);
      V[i * D + d] = static_cast<float>(i + d);
    }
  cpu_reference(Q, K, V, Oref, N, D, scale);

  float *dQ, *dK, *dV, *dS, *dP, *dO;
  CUDA_CHECK(cudaMalloc(&dQ, sizeof(float) * N * D));
  CUDA_CHECK(cudaMalloc(&dK, sizeof(float) * N * D));
  CUDA_CHECK(cudaMalloc(&dV, sizeof(float) * N * D));
  CUDA_CHECK(cudaMalloc(&dS, sizeof(float) * N * N));
  CUDA_CHECK(cudaMalloc(&dP, sizeof(float) * N * N));
  CUDA_CHECK(cudaMalloc(&dO, sizeof(float) * N * D));
  CUDA_CHECK(cudaMemcpy(dQ, Q.data(), sizeof(float) * N * D,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK, K.data(), sizeof(float) * N * D,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, V.data(), sizeof(float) * N * D,
                        cudaMemcpyHostToDevice));

  const dim3 block(kTile, kTile);
  const dim3 grid2d((N + kTile - 1) / kTile, (N + kTile - 1) / kTile);
  const dim3 grid1d((N + kTile - 1) / kTile, (D + kTile - 1) / kTile);
  attn_scores_tiled<<<grid2d, block>>>(dQ, dK, dS, N, D, scale);
  CUDA_CHECK(cudaGetLastError());
  row_softmax_stable<<<(N + 255) / 256, 256>>>(dS, dP, N);
  CUDA_CHECK(cudaGetLastError());
  attn_apply_tiled<<<grid1d, block>>>(dP, dV, dO, N, D);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(O.data(), dO, sizeof(float) * N * D,
                        cudaMemcpyDeviceToHost));

  float max_err = 0.0f;
  for (int i = 0; i < N * D; ++i) {
    max_err = std::max(max_err, std::fabs(O[i] - Oref[i]));
  }
  const long payload = 4L * N * N + 4L * N * D;  // modeled fp32 elements
  std::cout << "N=" << N << " D=" << D
            << " max|gpu - cpu|=" << max_err
            << " modeled_baseline_hbm_payload_fp32_elem=" << payload
            << std::endl;
  std::cout << "NOTE: baseline materializes S (" << N << "x" << N
            << "); fused online-softmax kernel not shipped -- "
            << "CUDA/H100 execution not validated." << std::endl;
  for (auto p : {dQ, dK, dV, dS, dP, dO}) CUDA_CHECK(cudaFree(p));
  return max_err < 1e-4 ? EXIT_SUCCESS : EXIT_FAILURE;
}
