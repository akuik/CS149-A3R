#ifndef _CUDA_RENDERER_H_
#define _CUDA_RENDERER_H_

#include "image.h"
#include "sceneLoader.h"
#include "circleRenderer.h"

class CudaRenderer : public CircleRenderer {

public:
    CudaRenderer();
    virtual ~CudaRenderer();

    virtual const Image* getImage();
    virtual void setup();
    virtual void loadScene(SceneName scene);
    virtual void allocOutputImage(int width, int height);
    virtual void clearImage();
    virtual void advanceAnimation();
    virtual void render();

private:

    Image* image;

    SceneName sceneName;
    int numCircles;

    float* position;
    float* velocity;
    float* color;
    float* radius;

    // Device pointers
    float* cudaDevicePosition;
    float* cudaDeviceVelocity;
    float* cudaDeviceColor;
    float* cudaDeviceRadius;
    float* cudaDeviceImageData;

    // Host data structures for tiled rendering
    int* circleTileLists;
    int* tileCircleCounts;
    int* tileCircleStartIndices;

    // Device data structures for tiled rendering
    int* cudaCircleTileList;
    int* cudaTileCircleCounts;
    int* cudaTileCircleStartIndices;

    // For tile calculations
    int totalTiles;
    int maxCirclesPerTile;

    // Tile dimensions
    int tileWidth;
    int tileHeight;
    int tilesPerRow;
    int tilesPerColumn;
    int imageWidth;
    int imageHeight;

    // Function to build circle-tile mapping
    void buildCircleTileLists();

    // Other helper functions (if any)
};

#endif
