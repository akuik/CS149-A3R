#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all CUDA
// kernels. The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU.
__constant__ GlobalConstants cuConstRendererParams;

// Read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// Color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];

#include "circleBoxTest.cu_inl"
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"

// Kernel to clear the image
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// Kernel to render pixels
__global__ void kernelRenderPixels() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    float invWidth = 1.f / width;
    float invHeight = 1.f / height;

    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(imageX) + 0.5f),
                                         invHeight * (static_cast<float>(imageY) + 0.5f));

    float4 pixelColor = make_float4(0.f, 0.f, 0.f, 0.f);

    // Process all circles in order
    for (int c = 0; c < cuConstRendererParams.numCircles; c++) {
        float3 p = *(float3*)(&cuConstRendererParams.position[3 * c]);
        float rad = cuConstRendererParams.radius[c];

        // Use circleInBoxConservative to quickly eliminate circles that can't contribute
        float circleCenterX = p.x;
        float circleCenterY = p.y;
        float circleRadius = rad;

        float boxL = pixelCenterNorm.x;
        float boxR = pixelCenterNorm.x;
        float boxB = pixelCenterNorm.y;
        float boxT = pixelCenterNorm.y;

        if (!circleInBoxConservative(circleCenterX, circleCenterY, circleRadius,
                                     boxL, boxR, boxT, boxB))
            continue;

        // Now perform the full point-in-circle test
        float diffX = p.x - pixelCenterNorm.x;
        float diffY = p.y - pixelCenterNorm.y;
        float pixelDist = diffX * diffX + diffY * diffY;
        float maxDist = rad * rad;

        if (pixelDist > maxDist)
            continue;

        // Compute shading and blend into pixelColor
        float3 rgb;
        float alpha;

        if (cuConstRendererParams.sceneName == SNOWFLAKES ||
            cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

            const float kCircleMaxAlpha = .5f;
            const float falloffScale = 4.f;

            float normPixelDist = sqrt(pixelDist) / rad;
            rgb = lookupColor(normPixelDist);

            float maxAlpha = .6f + .4f * (1.f - p.z);
            maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
            alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

        } else {
            int index3 = 3 * c;
            rgb = *(float3*)&(cuConstRendererParams.color[index3]);
            alpha = .5f;
        }

        float oneMinusAlpha = 1.f - alpha;

        pixelColor.x = alpha * rgb.x + oneMinusAlpha * pixelColor.x;
        pixelColor.y = alpha * rgb.y + oneMinusAlpha * pixelColor.y;
        pixelColor.z = alpha * rgb.z + oneMinusAlpha * pixelColor.z;
        pixelColor.w = alpha + oneMinusAlpha * pixelColor.w;
    }

    // Write the final color to the image
    int offset = 4 * (imageY * width + imageX);
    *(float4*)(&cuConstRendererParams.imageData[offset]) = pixelColor;
}

// Other kernels and functions remain the same...

////////////////////////////////////////////////////////////////////////////////////////

CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");

    // By this time the scene should be loaded. Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory
    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // Copy over the noise lookup tables
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // Copy over the color table
    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    cudaDeviceSynchronize();
}

// advanceAnimation --
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // Only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
    }
    cudaDeviceSynchronize();
}

void
CudaRenderer::render() {

    // Launch kernelRenderPixels with 16x16 threads per block
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    kernelRenderPixels<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
}

