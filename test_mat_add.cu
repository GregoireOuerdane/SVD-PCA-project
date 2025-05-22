#include <iostream>
#include <cuda_runtime.h>
#include "mat_add.cu"

int main() {
    const int rows = 4;
    const int cols = 4;
    const int size = rows * cols;
    float *h_A = new float[size];
    float *h_B = new float[size];
    float *h_C = new float[size];

    for (int i = 0; i < size; ++i) {
        h_A[i] = i;
        h_B[i] = 100 + i;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));

    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);
    mat_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
