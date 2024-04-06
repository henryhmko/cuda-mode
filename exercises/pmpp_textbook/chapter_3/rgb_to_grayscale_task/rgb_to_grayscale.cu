#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 16
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__
void rgbToGrayscaleKernel(unsigned char * Pout,
                unsigned char * Pin, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        // Get 1D offset for the grayscale image
        int grayOffset = row*width + col; // row-major
        // One can think of RGB image having CHANNEL
        // times more columns than the gray scale image
        int channels = 3;
        int rgbOffset = grayOffset * channels;
        unsigned char r = Pin[rgbOffset]; // Red
        unsigned char g = Pin[rgbOffset + 1]; // Green
        unsigned char b = Pin[rgbOffset + 2]; // Blue
        // Perform the rescaling and store the value
        Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

void rgbToGrayscale(unsigned char *inputImage, 
                    unsigned char *outputImage,
                    int width, int height) {
    unsigned char *d_inputImage, *d_outputImage;
    int imageSize = width * height * 3;
    int outputImageSize = width * height;

    // Allocate device memory
    cudaMalloc((void **) &d_inputImage, imageSize);
    cudaMalloc((void **) &d_outputImage, outputImageSize);

    // Copy input image from host to device
    cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);
    
    // Set grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); // Initialize a 2D block (16x16 in our case)
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y); // Understand this part more
    
    // Launch the kernel
    rgbToGrayscaleKernel<<<gridDim, blockDim>>>(d_outputImage, d_inputImage, width, height);

    // Copy output image from device to host
    cudaMemcpy(outputImage, d_outputImage, outputImageSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}



int main() {
    int width = 1024;
    int height = 768;

    // Allocate host memory for input and output images
    unsigned char *inputImage = (unsigned char *)malloc(width * height * 3);
    unsigned char *outputImage = (unsigned char *)malloc(width * height);
    
    // Initialize input image with random pixel values
    srand(time(NULL)); // Different random seed set for everytime we run
    for (int i = 0; i < width * height * 3; i++) {
        inputImage[i] = rand() % 256;
    }

    // Convert RGB image to grayscale
    rgbToGrayscale(inputImage, outputImage, width, height);

    // Save the img as png
    stbi_write_png("original_img.png", width, height, 3, inputImage, width);
    stbi_write_png("grayscale_img.png", width, height, 1, outputImage, width);
    printf("Original image saved as 'original_img.png'\n");
    printf("Grasycale image saved as 'grayscale_img.png'\n");
    
    // Free memory (host)
    free(inputImage);
    free(outputImage);

    return 0;
}