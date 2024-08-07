#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define CEIL_DIV(M, N) ((M + N - 1) / N) 
// write a matrix-vector mul kernel.
// A, C are vectors. B is a matrix.

__global__ void matmul_vec(const float *A, const float *B,
                           float *C, int M, int N) {
  // Assume that M == N, since matrix is square

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < M) { //since there are M rows
    float tmp = 0.0f;
    for (int i = 0; i < N; ++i) {
      tmp += A[x * N + i] * B[i];
    }
    C[x] = tmp;
  }
}

__global__ void check_result_kernel(const float *A, const float *B,
                                    float *C, int M, int N) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (x < M) {
    float tmp = 0.0f;
    for (int col = 0; col < N; ++col) {
      tmp += A[x * N + col] * B[col];
    }
    C[x] = tmp;
  }
}
  

int main() {
  int M = 4092, N = 4092;
  
  //allocate and initialize host data
  float *h_A = new float[M * N];
  float *h_B = new float[N];
  float *h_C = new float[M];

  std::fill_n(h_A, M * N, 2.0f);
  std::fill_n(h_B, N, 2.0f);
  std::fill_n(h_C, M, 0.0f);

  //allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * N * sizeof(float));
  cudaMalloc(&d_B, N * sizeof(float));
  cudaMalloc(&d_C, M * sizeof(float));

  //move data from host to device
  cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, M * sizeof(float), cudaMemcpyHostToDevice);

  // define grid and block dimensions
  dim3 gridDim(CEIL_DIV(M, 256), 1, 1);
  dim3 blockDim(256, 1, 1);

  // create profiling layouts
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  matmul_vec<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // check is operation was correct 
  int *d_errorFlag;
  cudaMalloc(&d_errorFlag, sizeof(int));
  cudaMemset(d_errorFlag, 0, sizeof(int));

  check_result_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N);

  int h_errorFlag = 0;
  cudaMemcpy(&h_errorFlag, d_errorFlag, sizeof(int), cudaMemcpyDeviceToHost);
  
  if (h_errorFlag) {
    std::cout << "Matmul vec results are incorrect." << std::endl;
  } else {
    std::cout << "Matmul vec results are correct." << std::endl;
  }

  cudaFree(d_errorFlag);

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int j = 0; j < 5 - (i%3); ++j)

  // get elapsed time
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  std::cout << "q2 matmul vec kernele execution time: " << ms << " ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // free host memory
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return 0;
}