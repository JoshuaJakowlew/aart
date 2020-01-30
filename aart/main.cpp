#include <iostream>
#include <chrono>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "Charmap.h"
#include "Comparators.h"

// Comprasion functions

template <typename T>
[[nodiscard]] auto create_art(cv::Mat& pic, const Charmap<T>& charmap) -> cv::Mat
{
	const auto cellw = charmap.cellW();
	const auto cellh = charmap.cellH();

	cv::resize(pic, pic, {}, 1.0, (double)cellw / cellh, cv::INTER_LINEAR);
	pic = convertTo<T>(pic);

	const auto picw = pic.size().width;
	const auto pich = pic.size().height;

	auto art = cv::Mat(pich * cellh, picw * cellw, charmap.type());

	pic.forEach<T>([&art, &charmap](auto p, const int* pos) noexcept {
		const auto y = pos[0];
		const auto x = pos[1];

		const auto cellw = charmap.cellW();
		const auto cellh = charmap.cellH();

		auto cell = charmap.getCell(p, CIE76_distance_sqr);
		const auto roi = cv::Rect{ x * cellw, y * cellh, cellw, cellh };
		cell.copyTo(art(roi));
	});

	return art;
}

template <typename T>
auto convert_video(const std::string& infile, const std::string& outfile, const Charmap<T>& charmap) -> void
{
	auto cap = cv::VideoCapture(infile, cv::CAP_FFMPEG);
	const int nframes = cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_COUNT);
	const int fps = cap.get(cv::VideoCaptureProperties::CAP_PROP_FPS);

	cv::Mat pic;
	cap >> pic;
	const auto art = create_art<T>(pic, charmap);

	const int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
	auto writer = cv::VideoWriter(outfile, cv::CAP_FFMPEG, fourcc, fps, art.size());

	writer << art;

	int frames_processed = 1;
	int frame_percent = nframes / 100;

	while (true)
	{
		cap >> pic;
		if (pic.empty())
			break;

		writer << art;//  create_art<T>(pic, charmap);

		if (++frames_processed % (frame_percent * 10) == 0)
			std::cout << frames_processed << '/' << nframes << " frames processed\n";
	}

	std::cout << "All frames processed\n";
}

template <typename T>
auto convert_image(const std::string& infile, const std::string& outfile, const Charmap<T>& charmap) -> void
{
	auto pic = cv::imread(infile);
	cv::imwrite(outfile, create_art<T>(pic, charmap));
}

int main(int argc, char* argv[])
{
	using namespace std::literals;
	using color_t = lab_t<float>;

	const auto charmap = Charmap<color_t>{
		cv::imread("charmap.png", cv::IMREAD_COLOR),
		cv::imread("colormap.png", cv::IMREAD_COLOR),
		" .:-=+*#%@"s
	};

	constexpr auto runs = 1;
	std::chrono::high_resolution_clock clock;
	auto start = clock.now();

	for (int i = 1; i <= runs; ++i)
	{
		convert_video<color_t>("test.mp4", "out.mp4", charmap);
	}

	auto end = clock.now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << duration / (double)runs << "ms avg in " << runs << " runs\n";
}
