#ifndef ART_H
#define ART_H

#include <fstream>

#include "charmap.h"
#include "comparators.h"
#include "cuda_kernels.h"

template <typename T, launch_t launch, distancef_t distance, mode_t mode>
class art_t {};

template <typename T>
class art_t<T, launch_t::cpu, distancef_t::CIE76, mode_t::image>
{
public:
	using lookup_table = charmap_t<T, launch_t::cpu, mode_t::image>;
	art_t(lookup_table charmap) :
		m_charmap{ std::move(charmap) }
	{}

	[[nodiscard]] auto create(cv::Mat& pic) const noexcept-> cv::Mat
	{
		cv::resize(pic, pic, {}, 1.0, (double)m_cellw / m_cellh, cv::INTER_LINEAR);
		pic = convert_to<T>(pic);

		const auto picw = pic.size().width;
		const auto pich = pic.size().height;

		auto art = cv::Mat(pich * m_cellh, picw * m_cellw, m_charmap.type());

		pic.forEach<T>([this, &art](auto p, const int* pos) noexcept {
			const auto y = pos[0];
			const auto x = pos[1];

			auto cell = m_charmap.getCell(p, CIE76_distance_sqr);
			const auto roi = cv::Rect{ x * m_cellw, y * m_cellh, m_cellw, m_cellh };
			cell.copyTo(art(roi));
		});

		return art;
	}
private:
	lookup_table m_charmap;
	const int m_cellw = m_charmap.cellW();
	const int m_cellh = m_charmap.cellH();
};

template <typename T>
class art_t<T, launch_t::cpu, distancef_t::CIE94, mode_t::image>
{
public:
	using lookup_table = charmap_t<T, launch_t::cpu, mode_t::image>;
	art_t(lookup_table charmap) :
		m_charmap{ std::move(charmap) }
	{}

	[[nodiscard]] auto create(cv::Mat& pic) const noexcept -> cv::Mat
	{
		cv::resize(pic, pic, {}, 1.0, (double)m_cellw / m_cellh, cv::INTER_LINEAR);
		pic = convert_to<T>(pic);

		const auto picw = pic.size().width;
		const auto pich = pic.size().height;

		auto art = cv::Mat(pich * m_cellh, picw * m_cellw, m_charmap.type());

		pic.forEach<T>([this, &art](auto p, const int* pos) noexcept {
			const auto y = pos[0];
			const auto x = pos[1];

			auto cell = m_charmap.getCell(p, CIE94_distance_sqr);
			const auto roi = cv::Rect{ x * m_cellw, y * m_cellh, m_cellw, m_cellh };
			cell.copyTo(art(roi));
			});

		return art;
	}
private:
	lookup_table m_charmap;
	const int m_cellw = m_charmap.cellW();
	const int m_cellh = m_charmap.cellH();
};

template <typename T>
class art_t<T, launch_t::cuda, distancef_t::CIE76, mode_t::image>
{
public:
	using lookup_table = charmap_t<T, launch_t::cuda, mode_t::image>;
	art_t(lookup_table charmap) :
		m_charmap{ std::move(charmap) }
	{}

	[[nodiscard]] auto create(cv::cuda::GpuMat& pic) const noexcept -> cv::cuda::GpuMat
	{
		cv::cuda::resize(pic, pic, {}, 1.0, (double)m_cellw / m_cellh, cv::INTER_LINEAR);
		pic = convert_to<T>(pic);

		const auto picw = pic.size().width;
		const auto pich = pic.size().height;

		auto art = cv::cuda::GpuMat(pich * m_cellh, picw * m_cellw, m_charmap.type());

		similarptr_t colors = similar2<distancef_t::CIE76>(pic, m_charmap.colormap());

		copy_symbols(
			art, m_charmap.charmap(), std::move(colors),
			picw, pich, m_charmap.cellW(), m_charmap.cellH(),
			m_charmap.ncolors(), m_charmap.nchars()
		);

		return art;
	}
private:
	lookup_table m_charmap;
	const int m_cellw = m_charmap.cellW();
	const int m_cellh = m_charmap.cellH();
};

template <typename T>
class art_t<T, launch_t::cuda, distancef_t::CIE94, mode_t::image>
{
public:
	using lookup_table = charmap_t<T, launch_t::cuda, mode_t::image>;
	art_t(lookup_table charmap) :
		m_charmap{ std::move(charmap) }
	{}

	[[nodiscard]] auto create(cv::cuda::GpuMat& pic) const noexcept -> cv::cuda::GpuMat
	{
		cv::cuda::resize(pic, pic, {}, 1.0, (double)m_cellw / m_cellh, cv::INTER_LINEAR);
		pic = convert_to<T>(pic);

		const auto picw = pic.size().width;
		const auto pich = pic.size().height;

		auto art = cv::cuda::GpuMat(pich * m_cellh, picw * m_cellw, m_charmap.type());

		similarptr_t colors = similar2<distancef_t::CIE94>(pic, m_charmap.colormap());

		copy_symbols(
			art, m_charmap.charmap(), std::move(colors),
			picw, pich, m_charmap.cellW(), m_charmap.cellH(),
			m_charmap.ncolors(), m_charmap.nchars()
			);

		return art;
	}
private:
	lookup_table m_charmap;
	const int m_cellw = m_charmap.cellW();
	const int m_cellh = m_charmap.cellH();
};

template <typename T>
class art_t<T, launch_t::cpu, distancef_t::CIE76, mode_t::ansi>
{
public:
	using lookup_table = charmap_t<T, launch_t::cpu, mode_t::ansi>;
	art_t(lookup_table charmap) :
		m_charmap{ std::move(charmap) }
	{}

	[[nodiscard]] auto create(cv::Mat& pic) const noexcept -> std::string
	{
		cv::resize(pic, pic, {}, 1.0, (double)m_cellw / m_cellh, cv::INTER_LINEAR);
		pic = convert_to<T>(pic);

		const auto picw = pic.size().width;
		const auto pich = pic.size().height;

		std::string art;
		for (int y = 0; y < pich; ++y)
		{
			for (int x = 0; x < picw; ++x)
			{
				const auto color = pic.at<T>(cv::Point2i{ x, y });
				art += m_charmap.getCell(color, CIE76_distance_sqr);
			}
			art += "\033[0m\n";
		}

		return art;
	}
private:
	lookup_table m_charmap;
	const int m_cellw = m_charmap.cellW();
	const int m_cellh = m_charmap.cellH();
};

template <typename T>
class art_t<T, launch_t::cpu, distancef_t::CIE94, mode_t::ansi>
{
public:
	using lookup_table = charmap_t<T, launch_t::cpu, mode_t::ansi>;
	art_t(lookup_table charmap) :
		m_charmap{ std::move(charmap) }
	{}

	[[nodiscard]] auto create(cv::Mat& pic) const noexcept -> std::string
	{
		cv::resize(pic, pic, {}, 1.0, (double)m_cellw / m_cellh, cv::INTER_LINEAR);
		pic = convert_to<T>(pic);

		const auto picw = pic.size().width;
		const auto pich = pic.size().height;

		std::string art;
		for (int y = 0; y < pich; ++y)
		{
			for (int x = 0; x < picw; ++x)
			{
				const auto color = pic.at<T>(cv::Point2i{ x, y });
				art += m_charmap.getCell(color, CIE94_distance_sqr);
			}
			art += "\033[0m\n";
		}

		return art;
	}
private:
	lookup_table m_charmap;
	const int m_cellw = m_charmap.cellW();
	const int m_cellh = m_charmap.cellH();
};

template <typename T>
auto convert_video(const std::string& infile, const std::string& outfile, const charmap_t<T, launch_t::cpu, mode_t::image>& charmap, distancef_t distance = distancef_t::CIE76) -> void
{
	auto cap = cv::VideoCapture(infile, cv::CAP_FFMPEG);
	const int nframes = cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_COUNT);
	const int fps = cap.get(cv::VideoCaptureProperties::CAP_PROP_FPS);

	art_t<T, launch_t::cpu, distancef_t::CIE76, mode_t::image> art_worker{ charmap };

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

template <typename T>
auto convert_image(const std::string& infile, const std::string& outfile, const charmap_t<T, launch_t::cpu, mode_t::image>& charmap, distancef_t distance = distancef_t::CIE76) -> void
{
	auto pic = cv::imread(infile);
	cv::imwrite(outfile, art_t<T, launch_t::cpu, distancef_t::CIE76, mode_t::image>{charmap}.create(pic));
}

#ifdef AART_CUDA
//template <typename T>
//[[nodiscard]] auto create_art(cv::cuda::GpuMat& pic, const charmap_t<T, launch_t::cuda>& charmap, distancef_t distance) -> cv::cuda::GpuMat
//{
//	const auto cellw = charmap.cellW();
//	const auto cellh = charmap.cellH();
//
//	cv::cuda::resize(pic, pic, {}, 1.0, (double)cellw / cellh, cv::INTER_LINEAR);
//	pic = convert_to<T>(pic);
//
//	const auto picw = pic.size().width;
//	const auto pich = pic.size().height;
//
//	auto art = cv::cuda::GpuMat(pich * cellh, picw * cellw, charmap.type());
//
//	similarptr_t colors = [distance, &pic, &charmap]() {
//		if (distance == distancef_t::CIE76)
//			return similar2<distancef_t::CIE76>(pic, charmap.colormap());
//		if (distance == distancef_t::CIE94)
//			return similar2<distancef_t::CIE94>(pic, charmap.colormap());
//	}();
//
//	copy_symbols(
//		art, charmap.charmap(), std::move(colors),
//		picw, pich, charmap.cellW(), charmap.cellH(),
//		charmap.ncolors(), charmap.nchars()
//	);
//
//	return art;
//}

template <typename T>
auto convert_image(const std::string& infile, const std::string& outfile, const charmap_t<T, launch_t::cuda, mode_t::image>& charmap, distancef_t distance = distancef_t::CIE76) -> void
{
	auto pic = cv::imread(infile);

	cv::cuda::GpuMat gpu_pic;
	gpu_pic.upload(pic);
	art_t<T, launch_t::cuda, distancef_t::CIE76, mode_t::image> art{ charmap };
	gpu_pic = art.create(gpu_pic);
	cv::Mat result;
	gpu_pic.download(result);

	cv::imwrite(outfile, result);
}

template <typename T>
auto convert_video(const std::string& infile, const std::string& outfile, const charmap_t<T, launch_t::cuda, mode_t::image>& charmap, distancef_t distance = distancef_t::CIE76) -> void
{
	auto cap = cv::VideoCapture(infile, cv::CAP_FFMPEG);
	const int nframes = cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_COUNT);
	const int fps = cap.get(cv::VideoCaptureProperties::CAP_PROP_FPS);

	cv::Mat pic;
	cap >> pic;

	art_t<T, launch_t::cpu, distancef_t::CIE76, mode_t::image> art_worker{ charmap };
	cv::cuda::GpuMat gpu_pic;
	gpu_pic.upload(pic);
	gpu_pic = art_worker.create(pic);
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
		gpu_pic = art_worker.create(pic);
		gpu_pic.download(art);

		writer << art;

		if (++frames_processed % (frame_percent * 10) == 0)
			std::cout << frames_processed << '/' << nframes << " frames processed\n";
	}

	std::cout << "All frames processed\n";
}
#endif // AART_CUDA
#endif

template <typename T>
auto convert_image(const std::string& infile, const std::string& outfile, const charmap_t<T, launch_t::cpu, mode_t::ansi>& charmap, distancef_t distance = distancef_t::CIE76) -> void
{
	auto pic = cv::imread(infile);
	std::ofstream fout(outfile);
	fout << art_t<T, launch_t::cpu, distancef_t::CIE76, mode_t::ansi>{charmap}.create(pic);
	fout.close();
}
