#include <cuda_runtime.h>
#include <iostream>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define TILE_WIDTH 32

__global__ void matmul_corner_turn_kernel(float *A, float *B, float *C,
                                          int width) {
  
  __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];


  // identify the row and col of the C element to work on
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

  // loop over A and B tiles required to compute C element
  float temp = 0.0f;
  for (int i = 0; i < width/TILE_WIDTH; ++i) {

    // collaborative loading of A and B tiles into shared memory
    Ads[threadIdx.y][threadIdx.x] = A[row*width + i*TILE_WIDTH + threadIdx.x];
    Bds[threadIdx.x][threadIdx.y] = B[(i*TILE_WIDTH + threadIdx.x)*width + col];
    __syncthreads();

   for (int k = 0; k < TILE_WIDTH; ++k) { // for each index within current tile
     temp += Ads[threadIdx.y][k] * Bds[threadIdx.x][k];
   }
   __syncthreads();
  }
  C[row*width + col] = temp;
}

__global__ void matmul_not_corner_turn_kernel(float *A, float *B, float *C,
                                          int width) {
  
  __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

  // identify the row and col of the C element to work on
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

  // loop over A and B tiles required to compute C element
  float temp = 0.0f;
  for (int i = 0; i < width/TILE_WIDTH; ++i) {

    // collaborative loading of A and B tiles into shared memory
    Ads[threadIdx.y][threadIdx.x] = A[row*width + i*TILE_WIDTH + threadIdx.x];
    Bds[threadIdx.y][threadIdx.x] = B[(i*TILE_WIDTH + threadIdx.y)*width + col];
    __syncthreads();

   for (int k = 0; k < TILE_WIDTH; ++k) { // for each index within current tile
     temp += Ads[threadIdx.y][k] * Bds[k][threadIdx.x];
   }
   __syncthreads();
  }
  C[row*width + col] = temp;
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
  int M = 4096, N = 4096, K = 4096;
  float alpha = 1.0f, beta = 0.0f;
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
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
  dim3 blockDim(32, 32, 1);

  // create profiling for kernel
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  matmul_corner_turn_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

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
  cudaEventElapsedTime(&ms, start, stop);
  std::cout<<"Corner turning kernel execution time: "<< ms << " ms" << std::endl;

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