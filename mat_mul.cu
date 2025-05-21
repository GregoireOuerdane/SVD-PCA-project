#include <iostream>
#include <cuda_runtime.h>

__global__ void mat_mul(const float *A, const float *B, float *C, int M, int N, int K){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < M && col < N){
      float sum = 0.0f;
      for (int i = 0; i < K; ++i){
        sum += A[row * K + i] * B[i * N + col];
      }
      C[row * N + col] = sum;
    }
}
