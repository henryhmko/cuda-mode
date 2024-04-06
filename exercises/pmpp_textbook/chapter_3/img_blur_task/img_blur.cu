#include <stdio.h>

#define BLOCK_SIZE 16
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLUR_SIZE 10 // avg of 10x10 box per pixel


__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    // Get Coordiantes
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is not out of bounds
    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;
        // Get average of the surrounding BLUR_SIZE * BLUR_SIZE box
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurRow;
                // Check if the cur pixel is not out of bounds
                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixVal += in[curRow*w + curCol]; // Because CUDA C is row-major
                    ++pixels;
                }
            }
        }
        out[row*w + col] = (unsigned char) (pixVal / pixels);
    }
}

void blur(unsigned char *inputImage, unsigned char *outputImage,
          int width, int height) {
    unsigned char *d_inputImage, *d_outputImage;
    int imageSize = width * height * 3;
    
    // Allocate device memory
    cudaMalloc((void **) &d_inputImage, imageSize);
    cudaMalloc((void **) &d_outputImage, imageSize);

    // Copy input image from host to device
    cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);
    
    // Set grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    blurKernel<<<gridDim, blockDim>>>(d_inputImage, d_outputImage, width, height);

    // Copy output image from device to host
    cudaMemcpy(outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

int main() {
    int width = 1024;
    int height = 768;

    // Allocate host memory for input and output images
    unsigned char *inputImage = (unsigned char *)malloc(width * height * 3);
    unsigned char *outputImage = (unsigned char *)malloc(width * height * 3);
    
    // Initialize input image with random pixel values
    srand(time(NULL));
    for (int i = 0; i < width * height * 3; i++) {
        inputImage[i] = rand() % 256;
    }

    // Blur image
    blur(inputImage, outputImage, width, height);

    // Save the img as png
    stbi_write_png("original_img.png", width, height, 3, inputImage, width);
    stbi_write_png("blurred_img.png", width, height, 3, outputImage, width);
    printf("Original image saved as 'original_img.png'\n");
    printf("Blurred image saved as 'blurred_img.png'\n");

    // Free host memory
    free(inputImage);
    free(outputImage);

    return 0;
}