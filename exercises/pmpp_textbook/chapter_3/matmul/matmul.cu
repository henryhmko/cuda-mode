#include <stdio.h>

#define BLOCK_SIZE 16

__global__
void matMulKernel(float* M, float* N,
                     float* P, int width) {
    // Get coordinates
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within bounds
    // Assuming both M and N are square matrices
    if ((row < width) && (col < width)) {
        float Pvalue = 0;
        for (int k = 0; k < width; k++) {
            Pvalue += M[row*width + k] * N[k*width + col];
        }
        P[row*width + col] = Pvalue;
    }
}

void matMul(float* M, float* N, float* P, int width) {
    float *d_M, *d_N, *d_P;
    int size = width * width * sizeof(float);

    cudaMalloc((void **) &d_M, size);
    cudaMalloc((void **) &d_N, size);
    cudamalloc((void **) &d_P, size);

    // from line 20 in vec add
    // copy memory
    // call kernel
    // copy mem back to host from device

    // cuda free
}