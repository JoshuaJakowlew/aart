#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <memory>

extern void add_with_cuda(const float* A, const float* B, float* C, int numElements);
extern void set_color_with_cuda(cv::InputArray input, cv::OutputArray output);

namespace cuda {
	using similar_t = SimilarColors<lab_t<float>, float>;
	extern auto similar2_CIE76_compare(cv::InputArray gpu_picture, cv::InputArray gpu_colormap) -> std::unique_ptr<similar_t>;
	extern auto copy_symbols(cv::cuda::GpuMat& gpu_art, const cv::cuda::GpuMat& gpu_charmap, const SimilarColors<lab_t<float>, float>* colors, int w, int h, int cellW, int cellH, int nChars) -> void;
	extern auto divide(cv::cuda::GpuMat& mat, float x) -> void;
}

#endif
