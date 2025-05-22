#include <iostream>
#include <cuda_runtime.h>
#include "mat_vec.cu"

int main() {
    const int rows = 4;
    const int cols = 4;
    const int size_matrix = rows * cols;
    const int size_vector = cols;
    const int size_result = rows;

    float h_A[size_matrix];
    float h_x[size_vector];
    float h_y[size_result];

    // Initialize matrix A as row-major: [[0,1,2,3],[4,5,6,7],...]
    for (int i = 0; i < size_matrix; ++i) {
        h_A[i] = static_cast<float>(i);
    }

    // Initialize vector x: [1, 2, 3, 4]
    for (int i = 0; i < size_vector; ++i) {
        h_x[i] = static_cast<float>(i + 1);
    }

    float *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, size_matrix * sizeof(float));
    cudaMalloc(&d_x, size_vector * sizeof(float));
    cudaMalloc(&d_y, size_result * sizeof(float));

    cudaMemcpy(d_A, h_A, size_matrix * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, size_vector * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16);
    dim3 blocksPerGrid((rows + 15) / 16);
    mat_vec<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_x, d_y, rows, cols);

    cudaMemcpy(h_y, d_y, size_result * sizeof(float), cudaMemcpyDeviceToHost);

    // Display results in vector form
    std::cout << "Result y = A * x:\n";
    for (int i = 0; i < rows; ++i) {
        std::cout << h_y[i] << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
