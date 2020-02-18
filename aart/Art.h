#ifndef ART_H
#define ART_H

#include "Charmap.h"
#include "Comparators.h"
#include "cuda_kernels.h"

template <typename T>
[[nodiscard]] auto create_art(cv::Mat& pic, const ñharmap_t<T, launch_t::cpu>& charmap) -> cv::Mat
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
auto convert_video(const std::string& infile, const std::string& outfile, const ñharmap_t<T, launch_t::cpu>& charmap) -> void
{
	auto cap = cv::VideoCapture(infile, cv::CAP_FFMPEG);
	const int nframes = cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_COUNT);
	const int fps = cap.get(cv::VideoCaptureProperties::CAP_PROP_FPS);

	cv::Mat pic;
	cap >> pic;
	const auto art = create_art<T>(pic, charmap);

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

		writer << create_art<T>(pic, charmap);

		if (++frames_processed % (frame_percent * 5) == 0)
			std::cout << frames_processed << '/' << nframes << " frames processed\n";
	}

	std::cout << "All frames processed\n";
}

template <typename T>
auto convert_image(const std::string& infile, const std::string& outfile, const ñharmap_t<T, launch_t::cpu>& charmap) -> void
{
	auto pic = cv::imread(infile);
	cv::imwrite(outfile, create_art<T>(pic, charmap));
}

template <typename T>
[[nodiscard]] auto create_art(cv::cuda::GpuMat& pic, const ñharmap_t<T, launch_t::cuda>& charmap) -> cv::cuda::GpuMat
{
	const auto cellw = charmap.cellW();
	const auto cellh = charmap.cellH();

	cv::cuda::resize(pic, pic, {}, 1.0, (double)cellw / cellh, cv::INTER_LINEAR);
	pic = convertTo<T>(pic);

	const auto picw = pic.size().width;
	const auto pich = pic.size().height;

	auto art = cv::cuda::GpuMat(pich * cellh, picw * cellw, charmap.type());

	auto colors = similar2_CIE76_compare(pic, charmap.colormap());
	copy_symbols(
		art, charmap.charmap(), std::move(colors),
		picw, pich, charmap.cellW(), charmap.cellH(),
		charmap.ncolors(), charmap.nchars()
	);

	return art;
}

template <typename T>
auto convert_image(const std::string& infile, const std::string& outfile, const ñharmap_t<T, launch_t::cuda>& charmap) -> void
{
	auto pic = cv::imread(infile);

	cv::cuda::GpuMat gpu_pic;
	gpu_pic.upload(pic);
	gpu_pic = create_art<T>(gpu_pic, charmap);
	cv::Mat art;
	gpu_pic.download(art);

	cv::imwrite(outfile, art);
}

template <typename T>
auto convert_video(const std::string& infile, const std::string& outfile, const ñharmap_t<T, launch_t::cuda>& charmap) -> void
{
	auto cap = cv::VideoCapture(infile, cv::CAP_FFMPEG);
	const int nframes = cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_COUNT);
	const int fps = cap.get(cv::VideoCaptureProperties::CAP_PROP_FPS);

	cv::Mat pic;
	cap >> pic;

	cv::cuda::GpuMat gpu_pic;
	gpu_pic.upload(pic);
	gpu_pic = create_art<T>(gpu_pic, charmap);
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
		gpu_pic = create_art<T>(gpu_pic, charmap);
		gpu_pic.download(art);

		writer << art;

		if (++frames_processed % (frame_percent * 10) == 0)
			std::cout << frames_processed << '/' << nframes << " frames processed\n";
	}

	std::cout << "All frames processed\n";
}

#endif
