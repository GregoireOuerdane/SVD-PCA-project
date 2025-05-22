#include <iostream>
#include <cuda_runtime.h>
#include "mat_mul.cu"

void print_matrix(const float* mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}

int main() {
    const int M = 4;  // rows of A and C
    const int K = 3;  // cols of A, rows of B
    const int N = 2;  // cols of B and C

    const int size_A = M * K;
    const int size_B = K * N;
    const int size_C = M * N;

    float h_A[size_A] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    };

    float h_B[size_B] = {
        1, 2,
        3, 4,
        5, 6
    };

    float h_C[size_C] = {0};

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_B, size_B * sizeof(float));
    cudaMalloc(&d_C, size_C * sizeof(float));

    cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);
    mat_mul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Matrix A:" << std::endl;
    print_matrix(h_A, M, K);
    std::cout << "\nMatrix B:" << std::endl;
    print_matrix(h_B, K, N);
    std::cout << "\nResult C = A * B:" << std::endl;
    print_matrix(h_C, M, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
