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

    // new variables: tiled rendering
    int* circleTileList;
    int* tileCircleCounts;
    int* tileCircleStartIndices;

    int tileWidth;
    int tileHeight;
    int tilesPerRow;
};

__constant__ GlobalConstants cuConstRendererParams;

// RO lookup tables (noise comp)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];

#include "circleBoxTest.cu_inl"
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"
// #include "exclusiveScan.cu_inl"


__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}


__global__ void kernelAdvanceBouncingBalls() {
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton v2
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}


__global__ void kernelAdvanceHypnosis() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}


__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { 
        // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}


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

            // circle processing (in order)
            for (int i = 0; i < numCirclesInTile; i++) {
                int c = cuConstRendererParams.circleTileList[circleStartIndex + i];

                float3 p = *(float3*)(&cuConstRendererParams.position[3 * c]);
                float rad = cuConstRendererParams.radius[c];

                // Use circleInBoxConservative() to eliminate/prune off circles
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

                // else (passed)
                float diffX = p.x - pixelCenterNorm.x;
                float diffY = p.y - pixelCenterNorm.y;
                float pixelDist = diffX * diffX + diffY * diffY;
                float maxDist = rad * rad;

                if (pixelDist > maxDist)
                    continue;


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

            // final color write
            *(float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]) = pixelColor;
        }
    }
}


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

    // palceholder: new vals
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

        // free created vals
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

    // need to copy contents of the rendered image from device memory
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
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

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

    // NOTE: tile dimensions set to (32 x 32)
    tileWidth = 32;
    tileHeight = 32;
    tilesPerRow = (params.imageWidth + tileWidth - 1) / tileWidth;
    tilesPerColumn = (params.imageHeight + tileHeight - 1) / tileHeight;
    totalTiles = tilesPerRow * tilesPerColumn;

    params.tileWidth = tileWidth;
    params.tileHeight = tileHeight;
    params.tilesPerRow = tilesPerRow;

    buildCircleTileLists();

    // copy tile data to device
    cudaMalloc(&cudaCircleTileList, sizeof(int) * totalTiles * maxCirclesPerTile);
    cudaMalloc(&cudaTileCircleCounts, sizeof(int) * totalTiles);
    cudaMalloc(&cudaTileCircleStartIndices, sizeof(int) * totalTiles);

    cudaMemcpy(cudaCircleTileList, circleTileLists, sizeof(int) * totalTiles * maxCirclesPerTile, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaTileCircleCounts, tileCircleCounts, sizeof(int) * totalTiles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaTileCircleStartIndices, tileCircleStartIndices, sizeof(int) * totalTiles, cudaMemcpyHostToDevice);

    params.circleTileList = cudaCircleTileList;
    params.tileCircleCounts = cudaTileCircleCounts;
    params.tileCircleStartIndices = cudaTileCircleStartIndices;

    // copy parameters to constant memory
    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // copy over the noise lookup tables
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // copy over the color table
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

    // init per-tile circle lists
    std::vector<std::vector<int>> tileCircleIndices(numTiles);

    for (int c = 0; c < numCircles; c++) {
        float3 p = *(float3*)(&position[3 * c]);
        float rad = radius[c];

        // bounding box compute (img coordinates)
        int minX = static_cast<int>(image->width * (p.x - rad));
        int maxX = static_cast<int>(image->width * (p.x + rad)) + 1;
        int minY = static_cast<int>(image->height * (p.y - rad));
        int maxY = static_cast<int>(image->height * (p.y + rad)) + 1;

        // boundary clamping
        minX = std::max(minX, 0);
        maxX = std::min(maxX, image->width - 1);
        minY = std::max(minY, 0);
        maxY = std::min(maxY, image->height - 1);

        // tile indices compute
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


void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);

    imageWidth = width;
    imageHeight = height;
}

void
CudaRenderer::clearImage() {
    // NOTE: REQ SETTING 256 THREADS per block (16 * 16)
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
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(tilesPerRow, tilesPerColumn);

    kernelRenderTiles<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
}

