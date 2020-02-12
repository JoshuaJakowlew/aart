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
	auto colormap = cv::imread("colormap.png", cv::IMREAD_COLOR);

	cv::cuda::GpuMat gpu_charmap;
	cv::cuda::GpuMat gpu_colormap;

	gpu_charmap.upload(charmap);
	gpu_colormap.upload(colormap);

	const auto charmap_ = cuda::Charmap<color_t>{
		gpu_charmap,
		gpu_colormap,
		" .:-=+*#%@"s
	};
	const auto charmap__ = Charmap<color_t>{
		charmap,
		colormap,
		" .:-=+*#%@"s
	};

	constexpr auto runs = 200;
	std::chrono::high_resolution_clock clock;
	std::vector<size_t> run_time;
	run_time.reserve(runs);
	for (int i = 1; i <= runs; ++i)
	{
		auto start = clock.now();
		//cuda::convert_video<color_t>("test.mp4", "out.mp4", charmap_);
		//convert_video<color_t>("test.mp4", "out.mp4", charmap__);
		//cuda::convert_image<color_t>("test.jpg", "out.png", charmap_);
		convert_image<color_t>("test.jpg", "out.png", charmap__);
		auto end = clock.now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		run_time.emplace_back(duration);
	}

	std::sort(run_time.begin(), run_time.end());
	double avg = 0;
	for (auto i : run_time)
		avg += i;

	std::cout << run_time[0] << "ms fastest in " << runs << " runs\n";
	std::cout << *(run_time.end() - 1) << "ms longest in " << runs << " runs\n";
	std::cout << size_t(avg / runs) << "ms avg in " << runs << " runs\n";
}
