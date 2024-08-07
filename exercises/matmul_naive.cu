#include <cuda_runtime.h>
#include <iostream>

// this is CUDA C++ version
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    // int K is the inner dimension of the matmul
    // compute position in C that this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // const int BLOCKSIZE = 1024;
    // const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    // const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    // // 'if' condition is necessary for when M or N aren't multiples of 32.
    if (x < M && y < N) { // is x and y within the bounds of the matrix?
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        // C = a*(A@B)+b*C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

int main() {
    // define M, N, K, alpha, beta (2 lines)
    int M = 4092, N = 4092, K = 4092;
    float alpha = 1.0f, beta = 0.0f;

    // allocate and initialize host matrices (6 lines)
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    // initialize
    std::fill_n(h_A, M * K, 2.0f);
    std::fill_n(h_B, K * N, 1.0f);
    std::fill_n(h_C, M * N, 0.0f);

    // allocate device memory (4 lines)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // move data from host to device (3 lines)
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // define grid and block dimensions (2 lines)
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    dim3 blockDim(32, 32, 1); // max TPB = 1024 for 2080ti

    // create profiling for kernel 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start timing
    cudaEventRecord(start);

    // launch kernel
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);

    //stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // get elpased time
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout<<"SGEMM kernel execution time: "<< ms << " ms" << std::endl;
    
    // destroy cuda events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
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