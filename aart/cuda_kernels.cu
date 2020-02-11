#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_kernels.h"

__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
    int i = threadIdx.x;
    
    C[i] = A[i] + B[i];
}

void add_with_cuda(const float* A, const float* B, float* C, int numElements)
{
    float *gpuA, *gpuB, *gpuC;
    cudaMalloc(&gpuA, sizeof(float) * numElements);
    cudaMalloc(&gpuB, sizeof(float) * numElements);
    cudaMalloc(&gpuC, sizeof(float) * numElements);

    cudaMemcpy(gpuA, A, sizeof(float) * numElements, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, B, sizeof(float) * numElements, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuC, C, sizeof(float) * numElements, cudaMemcpyHostToDevice);

    vectorAdd<<<1, numElements>>>(gpuA, gpuB, gpuC, numElements);

    cudaMemcpy(C, gpuC, sizeof(float) * numElements, cudaMemcpyDeviceToHost);
}