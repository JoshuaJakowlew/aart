#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <memory>

using similar_t = SimilarColors<float>;
using similarptr_t = std::unique_ptr<similar_t, void(*)(similar_t*)>;

enum class cuda_distancef_t
{
	CIE76,
	CIE94
};

extern auto similar2_CIE76(const cv::cuda::GpuMat& picture, const cv::cuda::GpuMat& colormap) -> similarptr_t;
extern auto similar2_CIE94(const cv::cuda::GpuMat& picture, const cv::cuda::GpuMat& colormap) -> similarptr_t;

template <cuda_distancef_t>
auto inline similar2(const cv::cuda::GpuMat& picture, const cv::cuda::GpuMat& colormap) -> similarptr_t
{}

template <>
auto inline similar2<cuda_distancef_t::CIE76>(const cv::cuda::GpuMat& picture, const cv::cuda::GpuMat& colormap) -> similarptr_t
{
	return similar2_CIE76(picture, colormap);
}

template <>
auto inline similar2<cuda_distancef_t::CIE94>(const cv::cuda::GpuMat& picture, const cv::cuda::GpuMat& colormap) -> similarptr_t
{
	return similar2_CIE94(picture, colormap);
}

extern auto copy_symbols(cv::cuda::GpuMat& art, const cv::cuda::GpuMat& charmap, const similarptr_t colors, int w, int h, int cellW, int cellH, int nColors, int nChars) -> void;
extern auto cuda_divide(cv::cuda::GpuMat& mat, float x) -> void;

#endif
