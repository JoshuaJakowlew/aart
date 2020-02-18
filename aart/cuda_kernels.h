#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <memory>

using similar_t = SimilarColors<float>;
using similarptr_t = std::unique_ptr<similar_t, void(*)(similar_t*)>;

extern auto similar2_CIE76_compare(const cv::cuda::GpuMat& picture, const cv::cuda::GpuMat& colormap) -> similarptr_t;
extern auto copy_symbols(cv::cuda::GpuMat& art, const cv::cuda::GpuMat& charmap, const similarptr_t colors, int w, int h, int cellW, int cellH, int nColors, int nChars) -> void;
extern auto cuda_divide(cv::cuda::GpuMat& mat, float x) -> void;

#endif
