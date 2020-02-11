#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

#include "Colors.h"
#include "cuda_kernels.h"

using similar_t = SimilarColors<lab_t<float>, float>;

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

__device__ float CIE76_compare(const lab_t<float>* x, const lab_t<float>* y)
{
    return pow(x->l - y->l, 2) + pow(x->a - y->a, 2) + pow(x->b - y->b, 2);
}

__global__ void similar2_CIE76_compare(const cv::cuda::PtrStepSz<lab_t<float>> picture, const cv::cuda::PtrStepSz<lab_t<float>> colormap, similar_t* similar)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= picture.cols - 1 && y <= picture.rows - 1 && y >= 0 && x >= 0)
    {
        const auto goal = picture(y, x);
        const auto start_color = colormap(0, 0);
        auto delta1 = CIE76_compare(&goal, &start_color);
        auto delta2 = delta1;
        auto color1 = start_color;
        auto color2 = color1;
        auto index1 = 0;
        auto index2 = index1;

        for (int i = 1; i < colormap.cols; ++i)
        {
            const auto color = colormap(0, i);
            const auto delta = CIE76_compare(&goal, &color);

            if (delta < delta1) {
                delta2 = delta1;
                delta1 = delta;

                color2 = color1;
                color1 = color;

                index2 = index1;
                index1 = i;
            }
            else if (delta < delta2) {
                delta2 = delta;

                color2 = color;

                index2 = i;
            }
        }

        similar[y * picture.cols + x] = similar_t{
                 color1,  color2,
                 delta1,  delta2,
                 index1,  index2
        };
    }
}

namespace cuda {
    [[nodiscard]] auto similar2_CIE76_compare(cv::InputArray gpu_picture, cv::InputArray gpu_colormap) -> std::unique_ptr<similar_t>
    {
        const auto picture = gpu_picture.getGpuMat();
        const auto colormap = gpu_colormap.getGpuMat();

        dim3 cthreads{ 16, 16 };
        dim3 cblocks{
            static_cast<unsigned>(std::ceil(picture.size().width /
                static_cast<double>(cthreads.x))),
            static_cast<unsigned>(std::ceil(picture.size().height /
                static_cast<double>(cthreads.y)))
        };

        similar_t* gpu_similar;

        cudaMalloc(&gpu_similar, sizeof(similar_t) * picture.rows * picture.cols);
        similar2_CIE76_compare<<<cblocks, cthreads>>>(picture, colormap, gpu_similar);

        auto err = cudaGetLastError();

        similar_t* similarp = (similar_t*)malloc(sizeof(similar_t) * picture.rows * picture.cols);
        auto similar = std::unique_ptr<similar_t>(similarp);

        cudaMemcpy(similar.get(), gpu_similar, sizeof(similar_t) * picture.rows * picture.cols, cudaMemcpyDeviceToHost);
        cudaFree(gpu_similar);
        return similar;
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