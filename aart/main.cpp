#include <iostream>
#include <chrono>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <CLI11.hpp>

#include "Art.h"

int main(int argc, char* argv[])
{
	using namespace std::literals;
	using color_t = lab_t<float>;
	constexpr auto ascii_grayscale = " .:-=+*#%@";

#pragma region parser_setup
	CLI::App app{
		"Convert images and videos to ascii-art!\nhttps://github.com/JoshuaJakowlew/aart"s,
		"Aart"s
	};

	std::string charmap_path;
	app.add_option("--chr,--charmap"s, charmap_path, "Path to the character map"s)->check(CLI::ExistingFile);

	std::string colormap_path;
	app.add_option("--clr,--colormap"s, colormap_path, "Path to the color map"s)->check(CLI::ExistingFile);

	std::string input_path;
	app.add_option("-i,--input"s, input_path, "Path to the input file"s)->check(CLI::ExistingFile);

	std::string output_path;
	app.add_option("-o,--output"s, output_path, "Path to the output file"s);

	int conv_mode{ 0 };
	app.add_flag("--img{0},--vid{1}"s, conv_mode, "Conversion mode [--img] for images, [--vid] for videos, [--img] if not specified"s);

	bool use_cuda{ false };
	app.add_flag("--cuda,!--no-cuda"s, use_cuda, "Use CUDA GPU acceleration (if possible). Better boost can be seen on videos, [--no-cuda] if not specified"s);
#pragma endregion parser_setup

#pragma region parsing_input
	try
	{
		app.parse(argc, argv);
	}
	catch (const CLI::ParseError& e)
	{
		return app.exit(e);
	}

	if (charmap_path == ""s || colormap_path == ""s || input_path == ""s || output_path == ""s)
	{
		std::cout << "Invalid input parameters.\nRun with --help for more information.\n";
		return EXIT_FAILURE;
	}
#pragma endregion parsing_input

	try
	{
		std::cout << "Conversion mode: " << (conv_mode == 0 ? "image" : "video")
				  << "\nUse CUDA: " << (use_cuda ? "yes" : "no")
				  << '\n';

		std::chrono::high_resolution_clock clock;
		auto start = clock.now();

		const auto cpu_charmap = cv::imread(charmap_path, cv::IMREAD_COLOR);
		const auto cpu_colormap = cv::imread(colormap_path, cv::IMREAD_COLOR);

		if (use_cuda)
		{
			cv::cuda::GpuMat gpu_charmap;
			cv::cuda::GpuMat gpu_colormap;

			gpu_charmap.upload(cpu_charmap);
			gpu_colormap.upload(cpu_colormap);

			const auto charmap = cuda::Charmap<color_t>{
				gpu_charmap,
				gpu_colormap,
				ascii_grayscale
			};

			if (conv_mode == 0)
			{
				cuda::convert_image<color_t>(input_path, output_path, charmap);
			}
			else if (conv_mode == 1)
			{
				cuda::convert_video<color_t>(input_path, output_path, charmap);
			}
		}
		else
		{
			const auto charmap = Charmap<color_t>{
				cpu_charmap,
				cpu_colormap,
				ascii_grayscale
			};

			if (conv_mode == 0)
			{
				convert_image<color_t>(input_path, output_path, charmap);
			}
			else if (conv_mode == 1)
			{
				convert_video<color_t>(input_path, output_path, charmap);
			}
		}

		auto end = clock.now();
		auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
		
		std::cout << "Elapsed time: " << duration << "s\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return EXIT_FAILURE;
	}
	catch (const cv::Exception & e)
	{
		std::cerr << e.what() << '\n';
		return EXIT_FAILURE;
	}
	catch (...)
	{
		std::cerr << "Unknown error\n";
		return EXIT_FAILURE;
	}
}
