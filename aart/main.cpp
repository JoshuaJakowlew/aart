#include <numeric>
#include <opencv2/opencv.hpp>

// Declare generic type for rgb color space

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
		using value_type   = rgb_t<T>;
		using work_type    = typename DataType<T>::work_type;
		using channel_type = T;

		enum {
			generic_type = 0,
			channels     = 3,
			fmt          = traits::SafeFmt<channel_type>::fmt + ((channels - 1) << 8)
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

// Declare generic type for lab color space

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

[[nodiscard]] auto normalize(cv::Mat& img) noexcept -> void
{
	img /= 255.0;
}

[[nodiscard]] auto bgr2lab(const cv::Mat& img) noexcept -> cv::Mat
{
	cv::Mat result;
	img.convertTo(result, CV_32F);
	normalize(result);
	cv::cvtColor(std::move(result), result, cv::COLOR_BGR2Lab);
	return result;
}

template <typename T>
[[nodiscard]] constexpr auto CIE76_distance(const lab_t<T>& x, const lab_t<T>& y) noexcept -> T
{
	return sqrt(pow(x.l - y.l, 2) + pow(x.a - y.a, 2) + pow(x.b - y.b, 2));
}

template <typename T, typename F>
[[nodiscard]] auto similar(const T& goal, const cv::Mat& palette, F distance) noexcept -> T
{
	return std::reduce(palette.begin<T>() + 1, palette.end<T>(), *palette.begin<T>(),
		[&goal, distance] (auto&& prev, auto&& curr) noexcept
		{
			const auto prevD = distance(goal, std::forward<decltype(prev)>(prev));
			const auto currD = distance(goal, std::forward<decltype(curr)>(curr));
			return currD < prevD ? curr : prev;
		}
	);

	/*auto dmin = distance(goal, *palette.begin<T>());
	auto result = *palette.begin<T>();

	std::for_each(palette.begin<T>() + 1, palette.end<T>(), [&](auto color) noexcept {
		const auto dcolor = distance(goal, color);
		if (dcolor < dmin)
		{
			dmin = dcolor;
			result = color;
		}
	});

	return result;*/
}

int main()
{
	auto palette = bgr2lab(cv::imread("palette.bmp", cv::IMREAD_COLOR));

	auto pic = cv::imread("test.jpg", cv::IMREAD_COLOR);
	auto orig = pic.clone();

	pic = bgr2lab(pic);
	pic.forEach<lab_t<float>>([&palette](lab_t<float>& p, const int* pos) {
		p = similar(p, palette, CIE76_distance<float>);
	});

	cv::cvtColor(pic, pic, cv::COLOR_Lab2BGR);

	cv::imshow("1", orig);
	cv::imshow("2", pic);
	cv::waitKey(0);
}
