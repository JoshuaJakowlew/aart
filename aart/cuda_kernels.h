#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <memory>

extern void add_with_cuda(const float* A, const float* B, float* C, int numElements);
extern void set_color_with_cuda(cv::InputArray input, cv::OutputArray output);

namespace cuda {
	extern auto similar2_CIE76_compare(cv::InputArray gpu_picture, cv::InputArray gpu_colormap) -> std::unique_ptr<SimilarColors<lab_t<float>, float>>;
}

#endif
