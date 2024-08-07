#include <cuda_runtime.h>
#include <iostream>

#define CEIL_DIV(M, N) (((M) + (N) - 1)) / N)

__global__ void matmul_square_kernel(float *A, float *B,
                                     float *C, int width) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((row < width) && (col < width)) {
    float cvalue = 0.0f;

    for (int k = 0; k < width; ++k) {
      cvalue += A[row * width + k] * B[k * width + col];
    }
    C[row * width + col] = cvalue;
  }
}