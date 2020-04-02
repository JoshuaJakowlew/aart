#ifndef CHARMAP_H
#define CHARMAP_H

#include <vector>
#include <sstream>

#include "colors.h"
#include "launch_type.h"
namespace detail {
	template <typename T, typename MatType>
	class charmap_base_t
	{
	public:
		charmap_base_t(MatType charmap, MatType colormap, std::string chars) :
			m_charmap{ std::move(charmap) },
			m_colormap{ convert_to<T>(std::move(colormap)) },
			m_chars{ std::move(chars) }
		{}

		charmap_base_t(const std::string& charmap, const std::string& colormap, const std::string chars) :
			m_charmap{ cv::imread(charmap, cv::IMREAD_COLOR) },
			m_colormap{ convert_to<T>(cv::imread(colormap, cv::IMREAD_COLOR)) },
			m_chars{ std::move(chars) }
		{}

#pragma region getters
		[[nodiscard]] inline auto cellW() const noexcept
		{
			return m_cellw;
		}

		[[nodiscard]] inline auto cellH() const noexcept
		{
			return m_cellh;
		}

		[[nodiscard]] inline auto size() const noexcept
		{
			return m_ncells;
		}

		[[nodiscard]] inline auto type() const noexcept
		{
			return m_charmap.type();
		}

		[[nodiscard]] inline auto chars() const noexcept
		{
			return m_chars;
		}

		[[nodiscard]] inline auto nchars() const noexcept
		{
			return m_nchars;
		}

		[[nodiscard]] inline auto ncolors() const noexcept
		{
			return m_ncolors;
		}
#pragma endregion getters

	protected:
		MatType m_charmap;
		MatType m_colormap;
		const std::string m_chars;

		const int m_nchars = m_chars.length();
		const int m_ncolors = m_colormap.size().width;

		const int m_cellw = m_charmap.size().width / m_nchars;
		const int m_cellh = m_charmap.size().height / (m_ncolors * m_ncolors);
		const int m_ncells = m_nchars * m_ncolors * m_ncolors;
	};

	template <typename T>
	class cpu_charmap_t : public charmap_base_t<T, cv::Mat>
	{
	public:
		using charmap_base_t<T, cv::Mat>::charmap_base_t;

		template <distancef_t distancef>
		[[nodiscard]] auto getCell(const T& color) const noexcept -> cv::Mat
		{
			const auto colors = similar2(color, getComparator<distancef>());

			// Calculate character index
			const int char_pos = colors.fg_delta == 0 ?
				this->m_nchars - 1 :
				colors.bg_delta * (this->m_nchars - 1) / colors.fg_delta;

			// Calculate cell position in charmap
			const auto cell_x = char_pos * this->m_cellw;
			const auto cell_y = (colors.bg_index * this->m_ncolors + colors.fg_index) * this->m_cellh;

			return this->m_charmap(cv::Rect{ cell_x, cell_y, this->m_cellw, this->m_cellh });
		}

	protected:
		template <distancef_t distancef>
		[[nodiscard]] constexpr auto getComparator() const noexcept -> float(*)(const lab_t<float>&, const lab_t<float>&)
		{
			if constexpr (distancef == distancef_t::CIE76)
				return CIE76_distance_sqr;
			else
				return CIE94_distance_sqr;
		}

		template <typename D>
		[[nodiscard]] auto similar2(const T& goal, D(*distance)(const T&, const T&)) const noexcept -> SimilarColors<D>
		{
			const auto start_color = this->m_colormap.begin<T>();
			auto delta1 = distance(goal, *start_color);
			auto delta2 = delta1;
			auto color1 = start_color;
			auto color2 = color1;

			for (auto color = start_color + 1; color != this->m_colormap.end<T>(); ++color)
			{
				const auto delta = distance(goal, *color);

				if (delta < delta1) {
					delta2 = delta1;
					delta1 = delta;

					color2 = color1;
					color1 = color;
				}
				else if (delta < delta2) {
					delta2 = delta;

					color2 = color;
				}
			}

			const int index1 = color1 - start_color;
			const int index2 = color2 - start_color;

			return {
				 delta1,  delta2,
				 index1,  index2
			};
		}
	};

	template <typename T>
	class gpu_charmap_t : public charmap_base_t<T, cv::cuda::GpuMat>
	{
	public:
		gpu_charmap_t(cv::cuda::GpuMat charmap, cv::cuda::GpuMat colormap, std::string chars) :
			charmap_base_t{ charmap, colormap, chars }
		{}

		gpu_charmap_t(const std::string& charmap, const std::string& colormap, const std::string chars)
		{
			this->m_chars = std::move(chars);

			const auto cpu_charmap = cv::imread(charmap, cv::IMREAD_COLOR);
			const auto cpu_colormap = cv::imread(colormap, cv::IMREAD_COLOR);

			this->m_charmap.upload(cpu_charmap);

			cv::cuda::GpuMat gpu_colormap;
			gpu_colormap.upload(cpu_colormap);
			this->m_colormap = convert_to<T>(gpu_colormap);

			this->m_nchars = this->m_chars.length();
			this->m_ncolors = this->m_colormap.cols;

			this->m_cellw = this->m_charmap.size().width / this->m_nchars;
			this->m_cellh = this->m_charmap.size().height / (this->m_ncolors * this->m_ncolors);
			this->m_ncells = this->m_nchars * this->m_ncolors * this->m_ncolors;
		}

		[[nodiscard]] inline auto charmap() const noexcept -> const cv::cuda::GpuMat&
		{
			return this->m_charmap;
		}

		[[nodiscard]] inline auto colormap() const noexcept -> const cv::cuda::GpuMat&
		{
			return this->m_colormap;
		}
	};

	template <typename T>
	class ansi_charmap_t : cpu_charmap_t<T>
	{
	public:
		ansi_charmap_t(cv::Mat charmap, cv::Mat colormap, std::string chars)
		{
			this->m_charmap = std::move(charmap);
			this->m_colormap = colormap;
			this->m_chars = std::move(chars);

			fillAnsiColors();
			this->m_colormap = convert_to<T>(this->m_colormap);
		}

		ansi_charmap_t(const std::string& charmap, const std::string& colormap, const std::string chars)
		{
			this->m_charmap = cv::imread(charmap, cv::IMREAD_COLOR);
			this->m_colormap = cv::imread(colormap, cv::IMREAD_COLOR);
			this->m_chars = std::move(chars);

			fillAnsiColors();
			this->m_colormap = convert_to<T>(this->m_colormap);
		}

		template <distancef_t distancef>
		[[nodiscard]] auto getCell(const T& color) const noexcept -> std::string
		{
			const auto colors = similar2(color, getComparator<distancef>());

			// Calculate character index
			const int char_pos = colors.fg_delta == 0 ?
				this->m_nchars - 1 :
				colors.bg_delta * (this->m_nchars - 1) / colors.fg_delta;

			// Calculate cell position in charmap
			const auto cell_x = char_pos;
			const auto cell_y = (colors.bg_index * this->m_ncolors + colors.fg_index);

			return m_ansi_colors[cell_y * this->m_nchars + cell_x];
		}
	private:
		std::vector<std::string> m_ansi_colors;

		auto fillAnsiColors() -> void
		{
			m_ansi_colors.reserve(this->m_ncells);
			std::ostringstream buffer;

			for (auto bg = this->m_colormap.begin<bgr_t<uint8_t>>(); bg != this->m_colormap.end<bgr_t<uint8_t>>(); ++bg)
				for (auto fg = this->m_colormap.begin<bgr_t<uint8_t>>(); fg != this->m_colormap.end<bgr_t<uint8_t>>(); ++fg)
					for (const auto c : this->m_chars)
					{
						buffer << "\033[38;2;" << (*fg).b
							<< ';' << (*fg).g
							<< ';' << (*fg).g
							<< ";48;2;" << (*bg).b
							<< ';' << (*bg).g
							<< ';' << (*fg).r
							<< 'm' << c;

						m_ansi_colors.emplace_back(buffer.str());
					}
		}
	};
}

template <typename T>
using cpu_charmap_t = detail::cpu_charmap_t<T>;

template <typename T>
using gpu_charmap_t = detail::gpu_charmap_t<T>;

template <typename T>
using ansi_charmap_t = detail::ansi_charmap_t<T>;

#endif