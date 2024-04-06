#include <stdio.h>
#include <math.h>

// Revision 1: C does not exist
__global__
void vecAddKernel(float* A, float* B, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        B[i] += A[i];
    }
}

void vecAdd(float* A, float* B, int n) {
    float *A_d, *B_d; // No C
    int size = n * sizeof(float);

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, n);

    cudaMemcpy(B, B_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
}

int main(void) {
    int n = 1 << 20; //1mil
    float *A, *B;
    
    int size = n * sizeof(float);
    
    A = (float *)malloc(size);
    B = (float *)malloc(size);

    for (int i = 0; i < n; i++) {
        A[i] = i;
        B[i] = 2*i;
    }

    vecAdd(A, B, n);

    return 0;
}

// End of Revision 1: C does not exist




// Original Version:

// __global__
// void vecAddKernel(float* A, float* B, float* C, int n) {
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     if (i < n) {
//         C[i] = A[i] + B[i];
//     }
// }

// void vecAdd(float* A, float* B, float* C, int n) {
//     float *A_d, *B_d, *C_d;
//     int size = n * sizeof(float);
    
//     cudaMalloc((void **) &A_d, size);
//     cudaMalloc((void **) &B_d, size);
//     cudaMalloc((void **) &C_d, size);

//     cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
//     cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
    
//     vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
    
//     cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

//     cudaFree(A_d);
//     cudaFree(B_d);
//     cudaFree(C_d);
// }

// int main(void) {
//     int n = 1 << 20;
//     float *A, *B, *C;
    
//     int size = n * sizeof(float);
//     A = (float *)malloc(size);
//     B = (float *)malloc(size);
//     C = (float *)malloc(size);

//     // Populate vectors A, B
//     for (int i = 0; i < n; i++) {
//         A[i] = i;
//         B[i] = 2*i;
//     }

//     vecAdd(A, B, C, n);

//     return 0;
    
//     // Allocate Unified Memory
//     // cudaMallocManaged(&A, N*sizeof(float));
//     // cudaMallocManaged(&B, N*sizeof(float));
//     // cudaMallocManaged(&C, N*sizeof(float));

//     // Initialize A
// }
