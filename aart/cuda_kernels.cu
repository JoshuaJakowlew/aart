#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

#include "Colors.h"
#include "cuda_kernels.h"

__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
    int i = threadIdx.x;
    
    C[i] = A[i] + B[i];
}

__global__ void setColor(const cv::cuda::PtrStepSz<lab_t<float>> input, cv::cuda::PtrStepSz<lab_t<float>> output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= input.cols - 1 && y <= input.rows - 1 && y >= 0 && x >= 0)
    {
        auto color = input(y, x);
        color.l = 60.32f;
        color.a = 98.24f;
        color.b = -60.84f;
        output(y, x) = color;
        //output(y, x) = input(y, x);
    }
}

void set_color_with_cuda(cv::InputArray _input, cv::OutputArray _output)
{
    const cv::cuda::GpuMat input = _input.getGpuMat();

    _output.create(input.size(), input.type());
    cv::cuda::GpuMat output = _output.getGpuMat();

    dim3 cthreads(16, 16);
    dim3 cblocks(
        static_cast<int>(std::ceil(input.size().width /
            static_cast<double>(cthreads.x))),
        static_cast<int>(std::ceil(input.size().height /
            static_cast<double>(cthreads.y))));
    setColor<<<cblocks, cthreads>>> (input, output);
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