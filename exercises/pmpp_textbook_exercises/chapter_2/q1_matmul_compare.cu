#include <cuda_runtime.h>
#include <iostream>

#include <cmath>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__global__ void matmul_row(int M, int N, int K, const float *A,
                           const float *B, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < M) {
    // each thread runs through a ROW of A and the ENTIRE B
    // float tmp = 0.0f;
    for (int col = 0; col < N; ++col) { //for every column in B
      float tmp = 0.0f;
      for (int i = 0; i < K; ++i) { //for every element in a row of A
        tmp += A[x * K + i] * B[i * N + col];
      }
      C[x * N + col] = tmp;
    }
  }
}

__global__ void matmul_col(int M, int N, int K, const float *A,
                           const float *B, float *C) {
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (col < N) {
    // each thread runs through a COL of B and the entire A
    for (int row = 0; row < M; ++row) { // for each row in A
      float tmp = 0.0f;
      for (int i = 0; i < K; ++i) { // for each element in col of B
        tmp += A[row * K + i] * B[i * N + col];
      }
      C[row * N + col] = tmp;
    }
  } 
}

__global__ void check_result_kernel(const float *A, const float*B, const float *C,
                                    int M, int N, int K, int *errorFlag) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < M && y < N) {
    float expected = 0.0f;
    for (int i = 0; i < K; ++i) {
      expected += A[x * K + i] * B[i * N + y];
    }
    float actual = C[x * N + y];
    if (fabsf(expected - actual) > 1e-5f) {
      atomicExch(errorFlag, 1); // vaue of errorFlag is 1; prevents race conditions
    }
  }
}

int main() {
  int M = 4092, N = 4092, K = 4092;
  // int M = 12288, N = 12288, K = 12288;

// allocate and initialize host matrices
  float *h_A = new float[M * K];
  float *h_B = new float[K * N];
  float *h_C = new float[M * N];

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

  // define grid and block dimensions
  dim3 gridDim_row(CEIL_DIV(M, 256), 1, 1);
  dim3 blockDim_row(256, 1, 1);

  dim3 gridDim_col(CEIL_DIV(N, 256), 1, 1);
  dim3 blockDim_col(256, 1, 1);

  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
  dim3 blockDim(32, 32, 1);

  // create profiling for kernel
  cudaEvent_t start_row, stop_row;
  cudaEventCreate(&start_row);
  cudaEventCreate(&stop_row);
  cudaEvent_t start_col, stop_col;
  cudaEventCreate(&start_col);
  cudaEventCreate(&stop_col);

  cudaEventRecord(start_col);
  matmul_col<<<gridDim_col, blockDim_col>>>(M, N, K, d_A, d_B, d_C);
  cudaEventRecord(stop_col);
  cudaEventSynchronize(stop_col);

  //override prev computations
  cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

  // start timing
  cudaEventRecord(start_row);
  // launch kernel
  matmul_row<<<gridDim_row, blockDim_row>>>(M, N, K, d_A, d_B, d_C);
  //stop timing
  cudaEventRecord(stop_row);
  cudaEventSynchronize(stop_row);

  

  // check is matmul was correct
  int *d_errorFlag;
  cudaMalloc(&d_errorFlag, sizeof(int));
  cudaMemset(d_errorFlag, 0, sizeof(int));

  check_result_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, d_errorFlag);

  int h_errorFlag = 0;
  cudaMemcpy(&h_errorFlag, d_errorFlag, sizeof(int), cudaMemcpyDeviceToHost);

  if (h_errorFlag) {
    std::cout << "Matmul results are incorrect." << std::endl;
  } else {
    std::cout << "Matmul results are correct." << std::endl;
  }

  cudaFree(d_errorFlag);

  // copy result back to host
  cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // get elpased time
  float ms = 0;
  cudaEventElapsedTime(&ms, start_row, stop_row);
  std::cout<<"q1_row_matmul row kernel execution time: "<< ms << " ms" << std::endl;

  float ms_col = 0;
  cudaEventElapsedTime(&ms_col, start_col, stop_col);
  std::cout<<"q1_col_matmul col kernel execution time: "<< ms_col << " ms" << std::endl;

  
  // destroy cuda events
  cudaEventDestroy(start_row);
  cudaEventDestroy(stop_row);
  cudaEventDestroy(start_col);
  cudaEventDestroy(stop_col);
  
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