#include <iostream>
#include <cuda_runtime.h>

__global__ void transpose(const float *A, float *A_T, int rows, int cols){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < rows && col < cols){
      int idxa = row * cols + col;
      int idxat = col * rows + row;

      A_T[idxat] = A[idxa];
    }
}
