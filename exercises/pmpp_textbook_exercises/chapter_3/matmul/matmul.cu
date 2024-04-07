#include <stdio.h>

#define BLOCK_SIZE 16

__global__
void matMulKernel(float* M, float* N, float* P, int width) {
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
    cudaMalloc((void **) &d_P, size);

    cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, P, size, cudaMemcpyHostToDevice);

    matMulKernel<<<ceil(width/256.0), 256>>>(d_M, d_N, d_P, width);

    cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

int main(void) {
    int width = 1 << 10;

    float *M, *N, *P;

    int size = width * width * sizeof(float);

    M = (float *)malloc(size);
    N = (float *)malloc(size);
    P = (float *)malloc(size);

    for (int i = 0; i < width*width; i++) {
        M[i] = i;
        N[i] = 2*i;
    }
    
    matMul(M, N, P, width);

    return 0;
}