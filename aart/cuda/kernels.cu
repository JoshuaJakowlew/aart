#include <cuda_runtime.h>
#include <cuda.h>

__global__ void kernel_absolute(float2* src, float* dst, int rows, int cols, int iStep, int oStep)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; //Row number
    int j = blockIdx.x * blockDim.x + threadIdx.x; //Column number

    if (i < rows && j < cols)
    {
        /* Compute linear index from 2D indices */
        int tidIn = i * iStep + j;
        int tidOut = i * oStep + j;

        /* Read input value */
        float2 input = src[tidIn];

        /* Calculate absolute value */
        float output = input.x * input.x + input.y * input.y;

        /* Write output value */
        dst[tidOut] = output;
    }
}


//namespace cuda {
//    void run_kernel(const cv::cuda::GpuMat& pic, cv::cuda::GpuMat& art, const Charmap<lab_t<float>>& charmap)
//    {
//        auto x = 42;
//    }
//}