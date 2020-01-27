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

		auto cell = charmap.getCell(p, RGB_euclidian_sqr);
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
	auto writer = cv::VideoWriter(outfile, fourcc, fps, art.size());

	writer << art;

	int frames_processed = 1;
	int frame_percent = nframes / 100;

	while (true)
	{
		cap >> pic;
		if (pic.empty())
			break;

		writer << create_art<T>(pic, charmap);

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
	using color_t = rgb_t<float>;

	if (argc >= 2 && argv[1] == "--help"s)
	{
		std::cout << "Usage: aart charmap colormap mode [-p for picture, -v for video] input output\n"
				  << "Example: aart charmap.png colormap.png -p image.png art.png\n";
	}
	else if (argc == 6)
	{
		const auto charmap = Charmap<color_t>{
		cv::imread(argv[1], cv::IMREAD_COLOR),
		cv::imread(argv[2], cv::IMREAD_COLOR),
		" .:-=+*#%@"s
		};

		const auto mode = argv[3];
		if (mode == "-p"s)
		{
			std::cout << "Converting picture " << argv[4] << " to ascii art!"s;
			convert_image<color_t>(argv[4], argv[5], charmap);
		}
		else if (mode == "-v"s)
		{
			std::cout << "Converting video " << argv[4] << " to ascii art!\nPlease, wait. Video conversion can take a lot of time\n"s;
			convert_video<color_t>(argv[4], argv[5], charmap);
		}
		else
		{
			std::cout << "Error: wrong input, try use --help\n"s;
		}
	}
	else
	{
		std::cout << "Error: wrong input, try use --help\n"s;
	}
}
