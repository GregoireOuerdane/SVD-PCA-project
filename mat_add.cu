#include <iostream>
#include <cuda_runtime.h>

__global__ void mat_add(const float *A, const float *B, float *C, int rows, int cols){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < rows && col < cols){
      int idx = row * cols + col;
      C[idx] = A[idx] + B[idx];
    }
}
