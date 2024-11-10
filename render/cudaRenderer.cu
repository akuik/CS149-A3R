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
// Putting all the CUDA kernels here
////////////////////////////////////////////////////////////////////////////////////////

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

    // New variables for tiled rendering
    int* circleTileList;
    int* tileCircleCounts;
    int* tileCircleStartIndices;

    int tileWidth;
    int tileHeight;
    int tilesPerRow;
};

__constant__ GlobalConstants cuConstRendererParams;

// Read-only lookup tables used to quickly compute noise
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// Color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];

#include "circleBoxTest.cu_inl"
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"
#include "exclusiveScan.cu_inl"

////////////////////////////////////////////////////////////////////////////////////////
// Kernels for advancing animation
////////////////////////////////////////////////////////////////////////////////////////

// Include the existing kernels for advancing animations here

// kernelAdvanceSnowflake
__global__ void kernelAdvanceSnowflake() {
    // [Include the implementation from the original code]
}

// kernelAdvanceBouncingBalls
__global__ void kernelAdvanceBouncingBalls() {
    // [Include the implementation from the original code]
}

// kernelAdvanceHypnosis
__global__ void kernelAdvanceHypnosis() {
    // [Include the implementation from the original code]
}

// kernelAdvanceFireWorks
__global__ void kernelAdvanceFireWorks() {
    // [Include the implementation from the original code]
}

////////////////////////////////////////////////////////////////////////////////////////
// Kernels for rendering
////////////////////////////////////////////////////////////////////////////////////////

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

// Kernel to clear the image for the snowflake background
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height - imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// Kernel to render tiles
__global__ void kernelRenderTiles() {

    int tileX = blockIdx.x;
    int tileY = blockIdx.y;

    int tileIndex = tileY * cuConstRendererParams.tilesPerRow + tileX;

    int tileWidth = cuConstRendererParams.tileWidth;
    int tileHeight = cuConstRendererParams.tileHeight;

    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;

    int startX = tileX * tileWidth;
    int startY = tileY * tileHeight;

    int endX = min(startX + tileWidth, imageWidth);
    int endY = min(startY + tileHeight, imageHeight);

    int numCirclesInTile = cuConstRendererParams.tileCircleCounts[tileIndex];
    int circleStartIndex = cuConstRendererParams.tileCircleStartIndices[tileIndex];

    for (int pixelY = startY + threadIdx.y; pixelY < endY; pixelY += blockDim.y) {
        for (int pixelX = startX + threadIdx.x; pixelX < endX; pixelX += blockDim.x) {

            float invWidth = 1.f / imageWidth;
            float invHeight = 1.f / imageHeight;

            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));

            float4 pixelColor;
            pixelColor = *(float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);

            // Process all circles in order
            for (int i = 0; i < numCirclesInTile; i++) {
                int c = cuConstRendererParams.circleTileList[circleStartIndex + i];

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

                    float normPixelDist = sqrtf(pixelDist) / rad;
                    rgb = lookupColor(normPixelDist);

                    float maxAlpha = .6f + .4f * (1.f - p.z);
                    maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
                    alpha = maxAlpha * expf(-1.f * falloffScale * normPixelDist * normPixelDist);

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
            *(float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]) = pixelColor;
        }
    }
}

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

    // Initialize new variables
    circleTileLists = NULL;
    tileCircleCounts = NULL;
    tileCircleStartIndices = NULL;

    cudaCircleTileList = NULL;
    cudaTileCircleCounts = NULL;
    cudaTileCircleStartIndices = NULL;

    totalTiles = 0;
    maxCirclesPerTile = 0;

    tileWidth = 0;
    tileHeight = 0;
    tilesPerRow = 0;
    tilesPerColumn = 0;
    imageWidth = 0;
    imageHeight = 0;
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

        // Free new device variables
        cudaFree(cudaCircleTileList);
        cudaFree(cudaTileCircleCounts);
        cudaFree(cudaTileCircleStartIndices);
    }

    if (circleTileLists) {
        delete[] circleTileLists;
        delete[] tileCircleCounts;
        delete[] tileCircleStartIndices;
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

    // Set tile dimensions
    tileWidth = 32;
    tileHeight = 32;
    tilesPerRow = (params.imageWidth + tileWidth - 1) / tileWidth;
    tilesPerColumn = (params.imageHeight + tileHeight - 1) / tileHeight;
    totalTiles = tilesPerRow * tilesPerColumn;

    params.tileWidth = tileWidth;
    params.tileHeight = tileHeight;
    params.tilesPerRow = tilesPerRow;

    // Build circle-tile mapping on the host
    buildCircleTileLists();

    // Copy tile data to device
    cudaMalloc(&cudaCircleTileList, sizeof(int) * totalTiles * maxCirclesPerTile);
    cudaMalloc(&cudaTileCircleCounts, sizeof(int) * totalTiles);
    cudaMalloc(&cudaTileCircleStartIndices, sizeof(int) * totalTiles);

    cudaMemcpy(cudaCircleTileList, circleTileLists, sizeof(int) * totalTiles * maxCirclesPerTile, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaTileCircleCounts, tileCircleCounts, sizeof(int) * totalTiles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaTileCircleStartIndices, tileCircleStartIndices, sizeof(int) * totalTiles, cudaMemcpyHostToDevice);

    params.circleTileList = cudaCircleTileList;
    params.tileCircleCounts = cudaTileCircleCounts;
    params.tileCircleStartIndices = cudaTileCircleStartIndices;

    // Copy parameters to constant memory
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

void CudaRenderer::buildCircleTileLists() {

    int numTiles = totalTiles;

    // Initialize per-tile circle lists
    std::vector<std::vector<int>> tileCircleIndices(numTiles);

    for (int c = 0; c < numCircles; c++) {
        float3 p = *(float3*)(&position[3 * c]);
        float rad = radius[c];

        // Compute bounding box in image coordinates
        int minX = static_cast<int>(image->width * (p.x - rad));
        int maxX = static_cast<int>(image->width * (p.x + rad)) + 1;
        int minY = static_cast<int>(image->height * (p.y - rad));
        int maxY = static_cast<int>(image->height * (p.y + rad)) + 1;

        // Clamp to image boundaries
        minX = std::max(minX, 0);
        maxX = std::min(maxX, image->width - 1);
        minY = std::max(minY, 0);
        maxY = std::min(maxY, image->height - 1);

        // Compute tile indices
        int tileMinX = minX / tileWidth;
        int tileMaxX = maxX / tileWidth;
        int tileMinY = minY / tileHeight;
        int tileMaxY = maxY / tileHeight;

        for (int ty = tileMinY; ty <= tileMaxY; ty++) {
            for (int tx = tileMinX; tx <= tileMaxX; tx++) {
                int tileIndex = ty * tilesPerRow + tx;
                tileCircleIndices[tileIndex].push_back(c);
            }
        }
    }

    // Build flat arrays for device use
    tileCircleCounts = new int[numTiles];
    tileCircleStartIndices = new int[numTiles];

    maxCirclesPerTile = 0;
    for (int i = 0; i < numTiles; i++) {
        tileCircleCounts[i] = tileCircleIndices[i].size();
        if (tileCircleCounts[i] > maxCirclesPerTile)
            maxCirclesPerTile = tileCircleCounts[i];
    }

    int totalCircles = maxCirclesPerTile * numTiles;

    circleTileLists = new int[totalCircles];

    for (int i = 0; i < numTiles; i++) {
        tileCircleStartIndices[i] = i * maxCirclesPerTile;
        int offset = tileCircleStartIndices[i];
        for (int j = 0; j < tileCircleCounts[i]; j++) {
            circleTileLists[offset + j] = tileCircleIndices[i][j];
        }
    }
}

// allocOutputImage --
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);

    imageWidth = width;
    imageHeight = height;
}

// clearImage --
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (imageWidth + blockDim.x - 1) / blockDim.x,
        (imageHeight + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // Only advance animation for certain scenes
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

    // Launch kernelRenderTiles with blocks per tile
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(tilesPerRow, tilesPerColumn);

    kernelRenderTiles<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
}

