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

template <typename T, typename F>
[[nodiscard]] auto similar2(const T& goal, const cv::Mat& palette, F distance) noexcept -> std::tuple<T, T, int, int>
{
	auto dmin1 = distance(goal, *palette.begin<T>());
	auto dmin2 = dmin1;
	auto c1 = palette.begin<T>();
	auto c2 = c1;

	
	for (auto c = palette.begin<T>() + 1; c != palette.end<T>(); ++c)
	{
		auto delta = distance(goal, *c);

		if (delta < dmin1) {
			dmin2 = dmin1;
			dmin1 = delta;

			c2 = c1;
			c1 = c;
		}
		else if (delta < dmin2) {
			dmin2 = delta;

			c2 = c;
		}
	}

	return std::make_tuple(*c1, *c2, c1 - palette.begin<T>(), c2 - palette.begin<T>());
}

int main()
{
	auto colormap = bgr2lab(cv::imread("colormap.png", cv::IMREAD_COLOR));
	auto charmap = cv::imread("charmap.png", cv::IMREAD_COLOR);

	constexpr auto n_chars = 10; // 10 in current palette " .:-=+*#%@"
	const auto n_colors = colormap.size().width;
	const auto char_w = charmap.size().width / n_chars;
	const auto char_h = charmap.size().height / (n_colors * n_colors);
	const auto n_charmap = n_chars * n_colors * n_colors;

	auto pic = cv::imread("test.jpg", cv::IMREAD_COLOR);
	cv::resize(pic, pic, {}, 1.0, (double)char_w / char_h, cv::INTER_LINEAR);
	auto art = cv::Mat(pic.size().height * char_h, pic.size().width * char_w, charmap.type());

	pic = bgr2lab(pic);
	pic.forEach<lab_t<float>>([&colormap, &charmap, &art, char_w, char_h, n_chars, n_colors](lab_t<float>& p, const int* pos) {
		auto [c1, c2, i1, i2] = similar2(p, colormap, CIE76_distance<float>);
		auto d1 = CIE76_distance(p, c1);
		auto d2 = CIE76_distance(p, c2);
		int ic = d2 == 0 ? n_chars - 1 : d1 / d2 * (n_chars - 1);

		auto y = pos[0];
		auto x = pos[1];

		auto sx = ic * char_w;
		auto sy = i1 * char_h * n_colors + i2 * char_h;

		auto symbol = charmap.colRange(sx, sx + char_w).rowRange(sy, sy + char_h);
		auto dst = art.colRange(x * char_w, (x + 1) * char_w).rowRange(y * char_h, (y + 1) * char_h);
		symbol.copyTo(dst);
	});

	cv::cvtColor(pic, pic, cv::COLOR_Lab2BGR);

	//cv::imshow("1", pic);
	//cv::imshow("2", art);
	cv::imwrite("result.png", art);
	//cv::waitKey(0);
}
