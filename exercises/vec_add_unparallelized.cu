#include <stdio.h>
#include <math.h>

// UNPARALLELIZED VERSION

// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main(void) {
    // Left-shift operator used to set N to 1*2^20, which is 1,048,576(approx 1mil)
    int N = 1 << 20;
    float *x, *y;

    // Allocate Unified Memory -- accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // Initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    
    // Run kernel on 1M elements on the GPU
    add<<<1, 1>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    printf("Max error: %f\n", maxError);

    // Free memory
    cudaFree(x);
    cudaFree(y);
    
    return 0; 
}


// Additional steps for Profiling:
// $ nvprof ./vec_add_unparallelized

// Time: 52.624ms on RTX 2080ti