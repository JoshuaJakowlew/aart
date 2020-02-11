#ifndef COLORS_H
#define COLORS_H

#include <opencv2/opencv.hpp>

template <typename T>
struct rgb_t
{
	using value_type = T;

	constexpr rgb_t(T r, T g, T b) noexcept :
		r{ r },
		g{ g },
		b{ b }
	{}

	T r{};
	T g{};
	T b{};
};

namespace cv {
	template<typename T>
	struct DataType<rgb_t<T>>
	{
		using value_type = rgb_t<T>;
		using work_type = typename DataType<T>::work_type;
		using channel_type = T;

		enum {
			generic_type = 0,
			channels = 3,
			fmt = traits::SafeFmt<channel_type>::fmt + ((channels - 1) << 8)
		};

		using vec_type = Vec<channel_type, channels>;
	};

	namespace traits {
		template<typename T>
		struct Depth<rgb_t<T>>
		{
			enum { value = Depth<T>::value };
		};

		template<typename T>
		struct Type<rgb_t<T>>
		{
			enum { value = CV_MAKETYPE(Depth<T>::value, 3) };
		};
	}
}

template <typename T>
struct lab_t
{
	using value_type = T;

	constexpr lab_t(T l, T a, T b) noexcept :
		l{ l },
		a{ a },
		b{ b }
	{}

	T l{};
	T a{};
	T b{};
};

namespace cv {
	template<typename T>
	struct DataType<lab_t<T>>
	{
		using value_type = lab_t<T>;
		using work_type = typename DataType<T>::work_type;
		using channel_type = T;

		enum {
			generic_type = 0,
			channels = 3,
			fmt = traits::SafeFmt<channel_type>::fmt + ((channels - 1) << 8)
		};

		using vec_type = Vec<channel_type, channels>;
	};

	namespace traits {
		template<typename T>
		struct Depth<lab_t<T>>
		{
			enum { value = Depth<T>::value };
		};

		template<typename T>
		struct Type<lab_t<T>>
		{
			enum { value = CV_MAKETYPE(Depth<T>::value, 3) };
		};
	}
}

template <typename T>
[[noreturn]] auto inline convertTo(const cv::Mat& img) noexcept -> cv::Mat
{
	//static_assert(false, "Unsupported color type");
}

template <>
[[nodiscard]] auto inline convertTo<rgb_t<uint8_t>>(const cv::Mat& img) noexcept -> cv::Mat
{
	return img;
}

template <>
[[nodiscard]] auto inline convertTo<rgb_t<int32_t>>(const cv::Mat& img) noexcept -> cv::Mat
{
	return img;
}

template <>
[[nodiscard]] auto inline convertTo<rgb_t<float>>(const cv::Mat& img) noexcept -> cv::Mat
{
	cv::Mat result;
	img.convertTo(result, CV_32FC3);
	result /= 255.f; // normalize
	return result;
}

template <>
[[nodiscard]] auto inline convertTo<rgb_t<double>>(const cv::Mat& img) noexcept -> cv::Mat
{
	cv::Mat result;
	img.convertTo(result, CV_64FC3);
	result /= 255.0; // normalize
	return result;
}

template <>
[[nodiscard]] auto inline convertTo<lab_t<float>>(const cv::Mat& img) noexcept -> cv::Mat
{
	cv::Mat result;
	img.convertTo(result, CV_32FC3);
	result /= 255.f; // normalize
	cv::cvtColor(std::move(result), result, cv::COLOR_BGR2Lab);
	return result;
}

template <>
[[nodiscard]] auto inline convertTo<lab_t<double>>(const cv::Mat& img) noexcept -> cv::Mat
{
	cv::Mat result;
	img.convertTo(result, CV_64FC3);
	result /= 255.0; // normalize
	cv::cvtColor(std::move(result), result, cv::COLOR_BGR2Lab);
	return result;
}

namespace cuda {
	template <typename T>
	[[noreturn]] auto inline convertTo(const cv::cuda::GpuMat& img) noexcept -> cv::cuda::GpuMat
	{
		//static_assert(false, "Unsupported color type");
	}

	template <>
	[[nodiscard]] auto inline convertTo<lab_t<float>>(const cv::cuda::GpuMat& img) noexcept -> cv::cuda::GpuMat
	{
		cv::cuda::GpuMat result;
		img.convertTo(result, CV_32FC3);
		cv::cuda::divide(result, cv::Scalar_<float>{255.f}, result);
		cv::cuda::cvtColor(std::move(result), result, cv::COLOR_BGR2Lab);
		return result;
	}
}

#endif
