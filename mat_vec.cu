#include <iostream>
#include <cuda_runtime.h>

__global__ void mat_vec(const float *A, const float *x, float *y, int rows, int cols){
    int row = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < rows){
      float sum = 0.0f;
      for (int col = 0; col < cols; ++col){
        int idx = row * cols + col;
        sum += A[idx] * x[col];
      }
      y[row] = sum;
    }
}
