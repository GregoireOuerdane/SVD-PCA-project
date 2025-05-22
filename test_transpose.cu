#include <iostream>
#include <cuda_runtime.h>
#include "transpose.cu"

void print_matrix(const float* mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << "\t";
        }
        std::cout << "\n";
    }
}

int main() {
    const int rows = 4;
    const int cols = 3;
    const int size = rows * cols;
    
    float* h_A = new float[size];
    float* h_A_T = new float[size];  // Transposed size will be cols * rows

    // Initialize matrix A with values 0..11
    for (int i = 0; i < size; ++i) {
        h_A[i] = i;
    }

    float *d_A, *d_A_T;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_A_T, size * sizeof(float));

    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);
    transpose<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_A_T, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_A_T, d_A_T, size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Original Matrix A (" << rows << "x" << cols << "):\n";
    print_matrix(h_A, rows, cols);
    std::cout << "\nTransposed Matrix A_T (" << cols << "x" << rows << "):\n";
    print_matrix(h_A_T, cols, rows); // Transposed dims

    delete[] h_A;
    delete[] h_A_T;
    cudaFree(d_A);
    cudaFree(d_A_T);

    return 0;
}
