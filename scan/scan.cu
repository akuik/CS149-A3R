#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256

// helper function to round an integer up to the next power of 2
static inline int64_t nextPow2(int64_t n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;  // (64-bit: N > ~4.1mil keeps failing otherwise)
    n++;
    return n;
}


__global__ void upsweep_kernel(int* data, int64_t n, int64_t two_d, int64_t two_dplus1) {
    int64_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t index = threadId * two_dplus1;

    if (index + two_dplus1 - 1 < n) {
        data[index + two_dplus1 - 1] += data[index + two_d - 1];
    }
}

__global__ void downsweep_kernel(int* data, int64_t n, int64_t two_d, int64_t two_dplus1) {
    int64_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t index = threadId * two_dplus1;

    if (index + two_dplus1 - 1 < n) {
        int t = data[index + two_d - 1];
        data[index + two_d - 1] = data[index + two_dplus1 - 1];
        data[index + two_dplus1 - 1] += t;
    }
}

// compute flags kernel for find_repeats
__global__ void compute_flags_kernel(int* device_input, int* device_flags, int length) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < length - 1) {
        device_flags[i] = (device_input[i] == device_input[i + 1]) ? 1 : 0;
    } else if (i == length - 1) {
        device_flags[i] = 0;
    }
}

// write output kernel for find_repeats
__global__ void write_output_kernel(int* device_flags, int* scan_result, int* device_output, int length) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < length && device_flags[i] == 1) {
        int pos = scan_result[i];
        device_output[pos] = i;
    }
}

// exclusive_scan
void exclusive_scan(int* input, int N, int* result) {
    int64_t n = nextPow2(N); 

    // inp -> res
    cudaMemcpy(result, input, N * sizeof(int), cudaMemcpyDeviceToDevice);

    if (n > N) {
        cudaMemset(result + N, 0, (n - N) * sizeof(int));
    }

    // upsweep phase
    for (int64_t two_d = 1; two_d < n; two_d *= 2) {
        int64_t two_dplus1 = 2 * two_d;

        int64_t num_threads = (n + two_dplus1 - 1) / two_dplus1;

        if (num_threads > 0) {
            int numBlocks = (int)((num_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            upsweep_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(result, n, two_d, two_dplus1);
            cudaDeviceSynchronize();
        }
    }

    // set last element to zero
    cudaMemset(result + n - 1, 0, sizeof(int));

    // downsweep
    for (int64_t two_d = n / 2; two_d >= 1; two_d /= 2) {
        int64_t two_dplus1 = 2 * two_d;

        int64_t num_threads = (n + two_dplus1 - 1) / two_dplus1;

        if (num_threads > 0) {
            int numBlocks = (int)((num_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            downsweep_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(result, n, two_d, two_dplus1);
            cudaDeviceSynchronize();
        }
    }
}

// cudaScan --
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above.
double cudaScan(int* inarray, int* end, int* resultarray) {
    int* device_result;
    int* device_input;
    int N = end - inarray;

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2
    int64_t rounded_length = nextPow2(N);

    cudaMalloc((void**)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void**)&device_input, sizeof(int) * rounded_length);

    // Initialize device_input and device_result with input data
    cudaMemcpy(device_input, inarray, N * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize elements beyond N to zero
    if (rounded_length > N) {
        cudaMemset(device_input + N, 0, (rounded_length - N) * sizeof(int));
    }

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, device_result, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_result);

    double overallDuration = endTime - startTime;
    return overallDuration;
}

// cudaScanThrust --
double cudaScanThrust(int* inarray, int* end, int* resultarray) {
    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration;
}

// find_repeats --
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
int find_repeats(int* device_input, int length, int* device_output) {
    int64_t n = nextPow2(length);

    // Allocate device_flags and scan_result
    int* device_flags;
    int* scan_result;

    cudaMalloc((void**)&device_flags, n * sizeof(int));
    cudaMalloc((void**)&scan_result, n * sizeof(int));

    // initialize device_flags and scan_result beyond length to zero
    if (n > length) {
        cudaMemset(device_flags + length, 0, (n - length) * sizeof(int));
        cudaMemset(scan_result + length, 0, (n - length) * sizeof(int));
    }

    // compute device_flags
    int numThreads = THREADS_PER_BLOCK;
    int numBlocks = (length + numThreads - 1) / numThreads;

    compute_flags_kernel<<<numBlocks, numThreads>>>(device_input, device_flags, length);
    cudaDeviceSynchronize();

    // Perform exclusive scan on device_flags and store in scan_result
    exclusive_scan(device_flags, length, scan_result);
    cudaDeviceSynchronize();

    // Get the total number of repeats
    int total_repeats = 0;
    int last_flag = 0;
    int last_scan = 0;

    cudaMemcpy(&last_flag, device_flags + length - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_scan, scan_result + length - 1, sizeof(int), cudaMemcpyDeviceToHost);

    total_repeats = last_scan + last_flag;

    // write out output
    numBlocks = (length + numThreads - 1) / numThreads;
    write_output_kernel<<<numBlocks, numThreads>>>(device_flags, scan_result, device_output, length);
    cudaDeviceSynchronize();

    // Free device_flags and scan_result
    cudaFree(device_flags);
    cudaFree(scan_result);

    return total_repeats;
}

// cudaFindRepeats --
// Timing wrapper around find_repeats.
double cudaFindRepeats(int* input, int length, int* output, int* output_length) {
    int* device_input;
    int* device_output;
    int64_t rounded_length = nextPow2(length);

    cudaMalloc((void**)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void**)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize elements beyond length to zero
    if (rounded_length > length) {
        cudaMemset(device_input + length, 0, (rounded_length - length) * sizeof(int));
    }

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();

    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // Set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, result * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime;
    return duration;
}

void printCudaInfo() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
