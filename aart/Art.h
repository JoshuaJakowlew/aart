#ifndef ART_H
#define ART_H

#include <fstream>

#include "charmap.h"
#include "comparators.h"
#include "cuda_kernels.h"

namespace detail {
	template <typename T, typename Charmap>
	class art_base_t {
	public:
		art_base_t(Charmap charmap) :
			m_charmap{ std::move(charmap) }
		{}
	protected:
		Charmap m_charmap;
		const int m_cellw = m_charmap.cellW();
		const int m_cellh = m_charmap.cellH();
	};

	template <typename T, distancef_t distancef = distancef_t::CIE94>
	class cpu_art_t : public art_base_t<T, cpu_charmap_t<T>>
	{
	public:
		[[nodiscard]] auto create(cv::Mat& pic) const noexcept-> cv::Mat
		{
			cv::resize(pic, pic, {}, 1.0, (double)this->m_cellw / this->m_cellh, cv::INTER_LINEAR);
			pic = convert_to<T>(pic);

			const auto picw = pic.size().width;
			const auto pich = pic.size().height;

			auto art = cv::Mat(pich * this->m_cellh, picw * this->m_cellw, this->m_charmap.type());

			pic.forEach<T>([this, &art](auto p, const int* pos) noexcept {
				const auto y = pos[0];
				const auto x = pos[1];

				auto cell = this->m_charmap.getCell<distancef>(p);
				const auto roi = cv::Rect{ x * this->m_cellw, y * this->m_cellh, this->m_cellw, this->m_cellh };
				cell.copyTo(art(roi));
				});

			return art;
		}
	};

	template <typename T, distancef_t distancef = distancef_t::CIE94>
	class gpu_art_t : public art_base_t<T, gpu_charmap_t<T>>
	{
	public:
		[[nodiscard]] auto create(cv::cuda::GpuMat& pic) const noexcept -> cv::cuda::GpuMat
		{
			cv::cuda::resize(pic, this->pic, {}, 1.0, (double)this->m_cellw / this->m_cellh, cv::INTER_LINEAR);
			pic = convert_to<T>(pic);

			const auto picw = pic.size().width;
			const auto pich = pic.size().height;

			auto art = cv::cuda::GpuMat(pich * this->m_cellh, picw * this->m_cellw, this->m_charmap.type());

			similarptr_t colors = similar2<distancef>(pic, this->m_charmap.colormap());

			copy_symbols(
				art, this->m_charmap.charmap(), std::move(colors),
				picw, pich, this->m_charmap.cellW(), this->m_charmap.cellH(),
				this->m_charmap.ncolors(), this->m_charmap.nchars()
				);

			return art;
		}
	};

	template <typename T, distancef_t distancef = distancef_t::CIE94>
	class ansi_art_t : public art_base_t<T, cpu_charmap_t<T>>
	{
	public:
		[[nodiscard]] auto create(cv::Mat& pic) const noexcept -> std::string
		{
			cv::resize(pic, pic, {}, 1.0, (double)this->m_cellw / this->m_cellh, cv::INTER_LINEAR);
			pic = convert_to<T>(pic);

			const auto picw = pic.size().width;
			const auto pich = pic.size().height;

			std::string art;
			for (int y = 0; y < pich; ++y)
			{
				for (int x = 0; x < picw; ++x)
				{
					const auto color = pic.at<T>(cv::Point2i{ x, y });
					art += this->m_charmap.getCell(color, CIE76_distance_sqr);
				}
				art += "\033[0m\n";
			}

			return art;
		}
	};
}

template <typename T, distancef_t distancef>
using cpu_art_t = detail::cpu_art_t<T>;

template <typename T, distancef_t distancef>
using gpu_art_t = detail::gpu_art_t<T>;

template <typename T, distancef_t distancef>
using ansi_art_t = detail::ansi_art_t<T>;

#endif
