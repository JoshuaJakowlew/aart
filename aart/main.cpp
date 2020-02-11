#include <iostream>
#include <chrono>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "Art.h"
#include "cuda_kernels.h"

int main(int argc, char* argv[])
{
	using namespace std::literals;
	using color_t = lab_t<float>;

	const auto charmap = cv::imread("charmap.png", cv::IMREAD_COLOR);
	const auto colormap = cv::imread("colormap.png", cv::IMREAD_COLOR);
	cv::cuda::GpuMat gpu_charmap;
	cv::cuda::GpuMat gpu_colormap;

	gpu_charmap.upload(charmap);
	gpu_colormap.upload(colormap);

	const auto charmap_ = cuda::Charmap<color_t>{
		gpu_charmap,
		gpu_colormap,
		" .:-=+*#%@"s
	};

	constexpr auto runs = 1;
	std::chrono::high_resolution_clock clock;
	auto start = clock.now();

	for (int i = 1; i <= runs; ++i)
	{
		cuda::convert_image<color_t>("test.jpg", "out.png", charmap_);
	}

	auto end = clock.now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << duration / (double)runs << "ms avg in " << runs << " runs\n";
}
