#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <memory>
#include <functional>

namespace cuda {
	using similar_t = SimilarColors<float>;
	extern auto similar2_CIE76_compare(cv::InputArray gpu_picture, cv::InputArray gpu_colormap) -> std::unique_ptr<similar_t, void(*)(similar_t*)>;
	extern auto copy_symbols(cv::cuda::GpuMat& gpu_art, const cv::cuda::GpuMat& gpu_charmap, const std::unique_ptr<similar_t, void(*)(similar_t*)> colors, int w, int h, int cellW, int cellH, int nChars) -> void;
	extern auto divide(cv::cuda::GpuMat& mat, float x) -> void;
}

#endif
