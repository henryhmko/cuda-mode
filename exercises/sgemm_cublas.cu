#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

extern "C" {
int main() {
    // define M, N, K, alpha, beta
    int M = 4092, N = 4092, K = 4092;
    float alpha = 1.0f, beta = 0.0f;

    // allocate and initialize host matrices
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    
    // initialize
    std::fill_n(h_A, M * K, 2.0f);
    std::fill_n(h_B, K * N, 1.0f);
    std::fill_n(h_C, M * N, 0.0f);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // move data from host to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // create profiling for kernel 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start timing
    cudaEventRecord(start);

    // perform matrix multiplication using cuBLAS
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
                &alpha, d_B, N, d_A, K, &beta, d_C, N);

    // stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // get elapsed time
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "cuBLAS SGEMM execution time: " << ms << " ms" << std::endl;

    // destroy cuda events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // destroy cuBLAS handle
    cublasDestroy(handle);

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
}