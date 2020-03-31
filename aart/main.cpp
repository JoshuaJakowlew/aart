#include <iostream>
#include <chrono>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <CLI11.hpp>

#include "art.h"

int main(int argc, char* argv[])
{
	using namespace std::literals;
	using color_t = lab_t<float>;
	constexpr auto ascii_grayscale = " .:-=+*#%@";


//#pragma region parser_setup
//	CLI::App app{
//		"Convert images and videos to ascii-art!\nhttps://github.com/JoshuaJakowlew/aart"s,
//		"Aart"s
//	};
//
//	std::string charmap_path;
//	app.add_option("--chr,--charmap"s, charmap_path, "Path to the character map"s)->check(CLI::ExistingFile);
//
//	std::string colormap_path;
//	app.add_option("--clr,--colormap"s, colormap_path, "Path to the color map"s)->check(CLI::ExistingFile);
//
//	std::string input_path;
//	app.add_option("-i,--input"s, input_path, "Path to the input file"s)->check(CLI::ExistingFile);
//
//	std::string output_path;
//	app.add_option("-o,--output"s, output_path, "Path to the output file"s);
//
//	int conv_mode{ 0 };
//	app.add_flag("--img{0},--vid{1}"s, conv_mode, "Conversion mode [--img] for images, [--vid] for videos, [--img] if not specified"s);
//
//#ifdef AART_CUDA
//	bool use_cuda{ false };
//	app.add_flag("--cuda,!--no-cuda"s, use_cuda, "Use CUDA GPU acceleration (if possible). Better boost can be seen on videos, [--no-cuda] if not specified"s);
//#endif // AART_CUDA
//
//	bool use_cie94{ false };
//	app.add_flag("--cie94,!--no-cie94"s, use_cie94, "Use more precise but more expensive algorithm, use default if not specified"s);
//#pragma endregion parser_setup
//
//#pragma region parsing_input
//	try
//	{
//		app.parse(argc, argv);
//	}
//	catch (const CLI::ParseError& e)
//	{
//		return app.exit(e);
//	}
//
//	if (charmap_path == ""s || colormap_path == ""s || input_path == ""s || output_path == ""s)
//	{
//		std::cout << "Invalid input parameters.\nRun with --help for more information.\n";
//		return EXIT_FAILURE;
//	}
//#pragma endregion parsing_input
//
//	try
//	{
//		std::cout << "Conversion mode: " << (conv_mode == 0 ? "image" : "video")
//#ifdef AART_CUDA
//				  << "\nUse CUDA: " << (use_cuda ? "yes" : "no")
//#endif // AART_CUDA
//				  << "\nUse CIE94: " << (use_cie94 ? "yes" : "no")
//				  << '\n';
//
//		std::chrono::high_resolution_clock clock;
//		auto start = clock.now();
//
//		const auto cpu_charmap = cv::imread(charmap_path, cv::IMREAD_COLOR);
//		const auto cpu_colormap = cv::imread(colormap_path, cv::IMREAD_COLOR);
//
//		const auto distancef = [use_cie94]() {
//			if (use_cie94)
//				return distancef_t::CIE94;
//			else
//				return distancef_t::CIE76;
//		}();
//
//#ifdef AART_CUDA
//		if (use_cuda)
//		{
//			const auto charmap = charmap_t<color_t, launch_t::cuda>{
//				charmap_path,
//				colormap_path,
//				ascii_grayscale
//			};
//
//			if (conv_mode == 0)
//			{
//				convert_image<color_t>(input_path, output_path, charmap, distancef);
//			}
//			else if (conv_mode == 1)
//			{
//				convert_video<color_t>(input_path, output_path, charmap, distancef);
//			}
//		}
//		else
//#endif // AART_CUDA
//		{
//			const auto charmap = charmap_t<color_t, launch_t::cpu>{
//				cpu_charmap,
//				cpu_colormap,
//				ascii_grayscale
//			};
//
//			if (conv_mode == 0)
//			{
//				convert_image<color_t>(input_path, output_path, charmap, distancef);
//			}
//			else if (conv_mode == 1)
//			{
//				convert_video<color_t>(input_path, output_path, charmap, distancef);
//			}
//		}
//
//		auto end = clock.now();
//		auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
//		
//		std::cout << "Elapsed time: " << duration << "s\n";
//	}
//	catch (const std::exception& e)
//	{
//		std::cerr << e.what() << '\n';
//		return EXIT_FAILURE;
//	}
//	catch (const cv::Exception & e)
//	{
//		std::cerr << e.what() << '\n';
//		return EXIT_FAILURE;
//	}
//	catch (...)
//	{
//		std::cerr << "Unknown error\n";
//		return EXIT_FAILURE;
//	}

	const auto charmap = charmap_t<color_t, launch_t::cuda>{
		"charmap.png",
		"colormap.png",
		ascii_grayscale
	};

	convert_image<color_t>("test.jpg", "out.png", charmap, distancef_t::CIE76);
}
