#ifndef COLORS_CUDA_H
#define COLORS_CUDA_H

#include "../Colors.h"

namespace cuda {
	template <typename T>
	[[noreturn]] auto inline convertTo(const cv::cuda::GpuMat& img) noexcept -> cv::cuda::GpuMat
	{
		static_assert(false, "Unsupported color type");
	}

	template <>
	[[nodiscard]] auto inline convertTo<lab_t<float>>(const cv::cuda::GpuMat& img) noexcept -> cv::cuda::GpuMat
	{
		cv::cuda::GpuMat result;
		img.convertTo(result, CV_32FC3);
		cv::cuda::divide(std::move(result), cv::Scalar_<float>{ 255.f }, result);
		cv::cuda::cvtColor(std::move(result), result, cv::COLOR_BGR2Lab);
		return result;
	}

	template <>
	[[nodiscard]] auto inline convertTo<rgb_t<uint8_t>>(const cv::cuda::GpuMat& img) noexcept -> cv::cuda::GpuMat
	{
		cv::cuda::GpuMat result;
		img.convertTo(result, CV_8UC3);
		return result;
	}
}

#endif
