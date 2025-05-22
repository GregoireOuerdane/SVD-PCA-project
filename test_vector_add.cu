#include <iostream>
#include <cuda_runtime.h>
#include "vector_add.cu"

int main() {
    const int N = 1024;
    float *a, *b, *c, *d_a, *d_b, *d_c;

    a = new float[N];
    b = new float[N];
    c = new float[N];

    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = 2 * i;
    }

    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    vector_add<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i)
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
