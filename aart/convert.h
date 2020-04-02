#ifndef CONVERT_H
#define CONVERT_H

#include "Art.h"

template <typename T = lab_t<uint8_t>, distancef_t distancef = distancef_t::CIE94>
auto convert_image(const std::string& infile, const std::string& outfile, const cpu_charmap_t<T>& charmap) -> void
{
	auto pic = cv::imread(infile);
	cv::imwrite(outfile, cpu_art_t<T, distancef>{charmap}.create(pic));
}

template <typename T = lab_t<uint8_t>, distancef_t distancef = distancef_t::CIE94>
auto convert_image(const std::string& infile, const std::string& outfile, const gpu_charmap_t<T>& charmap) -> void
{
	auto pic = cv::imread(infile);

	cv::cuda::GpuMat gpu_pic;
	gpu_pic.upload(pic);
	gpu_pic = gpu_art_t<T, distancef>(charmap).create(gpu_pic);
	cv::Mat result;
	gpu_pic.download(result);

	cv::imwrite(outfile, result);
}

template <typename T = lab_t<uint8_t>, distancef_t distancef = distancef_t::CIE94>
auto convert_image(const std::string& infile, const std::string& outfile, const ansi_charmap_t<T>& charmap) -> void
{
	auto pic = cv::imread(infile);
	std::ofstream fout(outfile);
	ansi_art_t<T, distancef> art(charmap);
	fout << art.create(pic);
	fout.close();
}

template <typename T = lab_t<uint8_t>, distancef_t distancef = distancef_t::CIE94>
auto convert_video(const std::string& infile, const std::string& outfile, const cpu_charmap_t<T>& charmap) -> void
{
	auto cap = cv::VideoCapture(infile, cv::CAP_FFMPEG);
	const int nframes = cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_COUNT);
	const int fps = cap.get(cv::VideoCaptureProperties::CAP_PROP_FPS);

	cpu_art_t<T, distancef> art_worker(charmap);

	cv::Mat pic;
	cap >> pic;
	const auto art = art_worker.create(pic);

	const int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
	auto writer = cv::VideoWriter(outfile, cv::CAP_MSMF, fourcc, fps, art.size());

	writer << art;

	int frames_processed = 1;
	int frame_percent = nframes / 100;

	while (true)
	{
		cap >> pic;
		if (pic.empty())
			break;

		writer << art_worker.create(pic);

		if (++frames_processed % (frame_percent * 5) == 0)
			std::cout << frames_processed << '/' << nframes << " frames processed\n";
	}

	std::cout << "All frames processed\n";
}

template <typename T = lab_t<uint8_t>, distancef_t distancef = distancef_t::CIE94>
auto convert_video(const std::string& infile, const std::string& outfile, const gpu_charmap_t<T>& charmap) -> void
{
	auto cap = cv::VideoCapture(infile, cv::CAP_FFMPEG);
	const int nframes = cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_COUNT);
	const int fps = cap.get(cv::VideoCaptureProperties::CAP_PROP_FPS);

	cv::Mat pic;
	cap >> pic;

	gpu_art_t<T, distancef> art_worker{ charmap };
	cv::cuda::GpuMat gpu_pic;
	gpu_pic.upload(pic);
	gpu_pic = art_worker.create(gpu_pic);
	cv::Mat art;
	gpu_pic.download(art);

	const int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
	auto writer = cv::VideoWriter(outfile, cv::CAP_MSMF, fourcc, fps, art.size());

	writer << art;

	int frames_processed = 1;
	int frame_percent = nframes / 100;

	while (true)
	{
		cap >> pic;
		if (pic.empty())
			break;

		gpu_pic.upload(pic);
		gpu_pic = art_worker.create(gpu_pic);
		gpu_pic.download(art);

		writer << art;

		if (++frames_processed % (frame_percent * 10) == 0)
			std::cout << frames_processed << '/' << nframes << " frames processed\n";
	}

	std::cout << "All frames processed\n";
}

#endif //CONVERT_H
