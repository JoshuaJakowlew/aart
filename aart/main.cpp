#include <iostream>
#include <chrono>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <CLI11.hpp>

#include "convert.h"
#include "color_quantization.h"

using color_t = lab_t<float>;
constexpr auto ascii_grayscale = " .:-=+*#%@";

void image_mode(const std::string& charmap, const std::string& colormap, const std::string& input, const std::string& output, bool use_cuda, bool use_cie94);
void video_mode(const std::string& charmap, const std::string& colormap, const std::string& input, const std::string& output, bool use_cuda, bool use_cie94);
void ansi_mode(const std::string& charmap, const std::string& colormap, const std::string& input, const std::string& output, bool use_cie94);

enum class quantization_t
{
	kmean,
	dominant
};

void palette_mode(const std::string& input, int colors, quantization_t quantization);

int main(int argc, char* argv[])
{
		using namespace std::literals;
	
		const std::string keys =
			"{h help usage ? |            | print this message                           }"
			"{clr colormap   |colormap.png| colormap                                     }"
			"{chr charmap    |charmap.png | charmap                                      }"
			"{i              |            | input file                                   }"
			"{o              |            | output file                                  }"
			"{mode           |image       | render mode [image,video,ansi,palette]       }"
			"{use_cuda       |false       | use cuda backend if possible                 }"
			"{cie94          |true        | use more precise but more expensive algorithm}"
			"{colors         |16          | number of colors in palette mode             }"
			"{quantization   |dominant    | color quantization algorithm [kmean,dominant]}";

		cv::CommandLineParser parser{ argc, argv, keys };

		const auto charmap = parser.get<std::string>("chr");
		const auto colormap = parser.get<std::string>("clr");

		const auto input = parser.get<std::string>("i");
		const auto output = parser.get<std::string>("o");

		const mode_t mode = [mode_s = parser.get<std::string>("mode")]() mutable {
			if ("image" == mode_s) return mode_t::image;
			if ("video" == mode_s) return mode_t::video;
			if ("ansi" == mode_s) return mode_t::ansi;
			if ("palette" == mode_s) return mode_t::palette;
			return mode_t::image;
		}();

		const bool use_cuda = parser.get<bool>("use_cuda");
		const bool use_cie94 = parser.get<bool>("cie94");

		if (mode == mode_t::image)
			image_mode(charmap, colormap, input, output, use_cuda, use_cie94);
		if (mode == mode_t::video)
			video_mode(charmap, colormap, input, output, use_cuda, use_cie94);
		if (mode == mode_t::ansi)
			ansi_mode(charmap, colormap, input, output, use_cie94);
		else
		{
			const int colors = parser.get<int>("colors");
			const quantization_t quantization = [quantization_s = parser.get<std::string>("quantization")]() mutable {
				if ("kmean" == quantization_s) return quantization_t::kmean;
				if ("dominant" == quantization_s) return quantization_t::dominant;
			}();
			palette_mode(input, colors, quantization);
		}
}

void image_mode(const std::string& charmap, const std::string& colormap, const std::string& input, const std::string& output, bool use_cuda, bool use_cie94)
{
	if (use_cuda)
	{
		auto art_charmap = gpu_charmap_t<color_t>{
			charmap,
			colormap,
			ascii_grayscale
		};

		if (use_cie94)
			convert_image<color_t, distancef_t::CIE94>(input, output, art_charmap);
		else
			convert_image<color_t, distancef_t::CIE76>(input, output, art_charmap);
	}
	else
	{
		auto art_charmap = cpu_charmap_t<color_t>{
			charmap,
			colormap,
			ascii_grayscale
		};

		if (use_cie94)
			convert_image<color_t, distancef_t::CIE94>(input, output, art_charmap);
		else
			convert_image<color_t, distancef_t::CIE76>(input, output, art_charmap);
	}
}

void video_mode(const std::string& charmap, const std::string& colormap, const std::string& input, const std::string& output, bool use_cuda, bool use_cie94)
{
	if (use_cuda)
	{
		auto art_charmap = gpu_charmap_t<color_t>{
			charmap,
			colormap,
			ascii_grayscale
		};

		if (use_cie94)
			convert_video<color_t, distancef_t::CIE94>(input, output, art_charmap);
		else
			convert_video<color_t, distancef_t::CIE76>(input, output, art_charmap);
	}
	else
	{
		auto art_charmap = cpu_charmap_t<color_t>{
			charmap,
			colormap,
			ascii_grayscale
		};

		if (use_cie94)
			convert_video<color_t, distancef_t::CIE94>(input, output, art_charmap);
		else
			convert_video<color_t, distancef_t::CIE76>(input, output, art_charmap);
	}
}

void ansi_mode(const std::string& charmap, const std::string& colormap, const std::string& input, const std::string& output, bool use_cie94)
{
	auto art_charmap = ansi_charmap_t<color_t>{
			charmap,
			colormap,
			ascii_grayscale
	};

	if (use_cie94)
		convert_image<color_t, distancef_t::CIE94>(input, output, art_charmap);
	else
		convert_image<color_t, distancef_t::CIE76>(input, output, art_charmap);
}

void palette_mode(const std::string& input, int colors, quantization_t quantization)
{
	auto img = cv::imread(input, cv::IMREAD_COLOR);

	if (quantization == quantization_t::kmean)
	{
		cv::Mat centers = kmean(img, colors);

		for (int i = 0; i < centers.rows; ++i)
		{
			auto x = centers.at<float>(i, 0);
			auto y = centers.at<float>(i, 1);
			auto c = cv::Point2f{ x, y };
			auto clr = img.at<bgr_t<uint8_t>>(c);
			std::cout << "(" << (int)clr.r << ", " << (int)clr.g << ", " << (int)clr.b << "),\n";
		}
	}
	else
	{
		const auto palette = dominant_colors(img, colors);
		for (auto c : palette)
			std::cout << "(" << (int)c[2] << ", " << (int)c[1] << ", " << (int)c[0] << "),\n";
	}
}
