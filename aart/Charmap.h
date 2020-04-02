#ifndef CHARMAP_H
#define CHARMAP_H

#include <vector>
#include <sstream>

#include "colors.h"
#include "launch_type.h"

template <typename T, launch_t = launch_t::cpu, mode_t = mode_t::ansi>
class charmap_t {};

template <typename T>
class charmap_t<T, launch_t::cpu, mode_t::image>
{
public:
	charmap_t(cv::Mat charmap, cv::Mat colormap, std::string chars) :
		m_charmap{ std::move(charmap) },
		m_colormap{ convert_to<T>(std::move(colormap)) },
		m_chars{ std::move(chars) }
	{}

	charmap_t(const std::string& charmap, const std::string& colormap, const std::string chars) :
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
#pragma endregion getters

	template <typename F>
	[[nodiscard]] auto getCell(const T& color, F distance) const noexcept -> cv::Mat;

private:
#pragma region members
	cv::Mat m_charmap;
	cv::Mat m_colormap;
	const std::string m_chars;

	const int m_nchars = m_chars.length();
	const int m_ncolors = m_colormap.size().width;

	const int m_cellw = m_charmap.size().width / m_nchars;
	const int m_cellh = m_charmap.size().height / (m_ncolors * m_ncolors);
	const int m_ncells = m_nchars * m_ncolors * m_ncolors;
#pragma endregion members

	template <typename D>
	[[nodiscard]] auto similar2(const T& goal, D(*distance)(const T&, const T&)) const noexcept -> SimilarColors<D>;
};

template <typename T>
template <typename F>
[[nodiscard]] auto charmap_t<T, launch_t::cpu, mode_t::image>::getCell(const T& color, F distance) const noexcept -> cv::Mat
{
	const auto colors = similar2(color, distance);

	// Calculate character index
	const int char_pos = colors.fg_delta == 0 ?
		m_nchars - 1 :
		colors.bg_delta * (m_nchars - 1) / colors.fg_delta;

	// Calculate cell position in charmap
	const auto cell_x = char_pos * m_cellw;
	const auto cell_y = (colors.bg_index * m_ncolors + colors.fg_index) * m_cellh;

	return m_charmap(cv::Rect{ cell_x, cell_y, m_cellw, m_cellh });
}

template <typename T>
template <typename D>
[[nodiscard]] auto charmap_t<T, launch_t::cpu, mode_t::image>::similar2(const T& goal, D(*distance)(const T&, const T&)) const noexcept -> SimilarColors<D>
{
	const auto start_color = m_colormap.begin<T>();
	auto delta1 = distance(goal, *start_color);
	auto delta2 = delta1;
	auto color1 = start_color;
	auto color2 = color1;

	for (auto color = start_color + 1; color != m_colormap.end<T>(); ++color)
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

#ifdef AART_CUDA
template <typename T>
class charmap_t<T, launch_t::cuda, mode_t::image>
{
public:
	charmap_t(cv::cuda::GpuMat charmap, cv::cuda::GpuMat colormap, std::string chars) :
		m_charmap{ std::move(charmap) },
		m_colormap{ convert_to<T>(std::move(colormap)) },
		m_chars{ std::move(chars) },
		m_nchars{ m_chars.length() },
		m_ncolors{ m_colormap.cols },
		m_cellw{ m_charmap.size().width / m_nchars },
		m_cellh{ m_charmap.size().height / (m_ncolors * m_ncolors) },
		m_ncells{ m_nchars * m_ncolors * m_ncolors }
	{}

	charmap_t(const std::string& charmap, const std::string& colormap, const std::string chars) :
		m_chars{ std::move(chars) }
	{
		const auto cpu_charmap = cv::imread(charmap, cv::IMREAD_COLOR);
		const auto cpu_colormap = cv::imread(colormap, cv::IMREAD_COLOR);

		m_charmap.upload(cpu_charmap);

		cv::cuda::GpuMat gpu_colormap;
		gpu_colormap.upload(cpu_colormap);
		m_colormap = convert_to<T>(gpu_colormap);

		m_nchars = m_chars.length();
		m_ncolors = m_colormap.cols;

		m_cellw = m_charmap.size().width / m_nchars;
		m_cellh = m_charmap.size().height / (m_ncolors * m_ncolors);
		m_ncells = m_nchars * m_ncolors * m_ncolors;
	}

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

	[[nodiscard]] inline auto charmap() const noexcept -> const cv::cuda::GpuMat&
	{
		return m_charmap;
	}

	[[nodiscard]] inline auto colormap() const noexcept -> const cv::cuda::GpuMat&
	{
		return m_colormap;
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

private:
#pragma region members
	cv::cuda::GpuMat m_charmap;
	cv::cuda::GpuMat m_colormap;
	const std::string m_chars;

	int m_nchars;
	int m_ncolors;

	int m_cellw;
	int m_cellh;
	int m_ncells;
#pragma endregion members
};
#endif // AART_CUDA
#endif

template <typename T>
class charmap_t<T, launch_t::cpu, mode_t::ansi>
{
public:
	charmap_t(cv::Mat charmap, cv::Mat colormap, std::string chars) :
		m_charmap{ std::move(charmap) },
		m_colormap{ colormap },
		m_chars{ std::move(chars) }
	{
		fillAnsiColors();
		m_colormap = convert_to<T>(m_colormap);
	}

	charmap_t(const std::string& charmap, const std::string& colormap, const std::string chars) :
		m_charmap{ cv::imread(charmap, cv::IMREAD_COLOR) },
		m_colormap{ cv::imread(colormap, cv::IMREAD_COLOR) },
		m_chars{ std::move(chars) }
	{
		fillAnsiColors();
		m_colormap = convert_to<T>(m_colormap);
	}

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
#pragma endregion getters

	template <typename F>
	[[nodiscard]] auto getCell(const T& color, F distance) const noexcept -> std::string;

private:
#pragma region members
	cv::Mat m_charmap;
	cv::Mat m_colormap;
	const std::string m_chars;

	std::vector<std::string> m_ansi_colors;

	const int m_nchars = m_chars.length();
	const int m_ncolors = m_colormap.size().width;

	const int m_cellw = m_charmap.size().width / m_nchars;
	const int m_cellh = m_charmap.size().height / (m_ncolors * m_ncolors);
	const int m_ncells = m_nchars * m_ncolors * m_ncolors;
#pragma endregion members
	auto fillAnsiColors() -> void
	{
		m_ansi_colors.reserve(m_ncells);
		std::ostringstream buffer;

		for (auto bg = m_colormap.begin<bgr_t<uint8_t>>(); bg != m_colormap.end<bgr_t<uint8_t>>(); ++bg)
			for (auto fg = m_colormap.begin<bgr_t<uint8_t>>(); fg != m_colormap.end<bgr_t<uint8_t>>(); ++fg)
				for (const auto c : m_chars)
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

	template <typename D>
	[[nodiscard]] auto similar2(const T& goal, D(*distance)(const T&, const T&)) const noexcept -> SimilarColors<D>;
};

template <typename T>
template <typename F>
[[nodiscard]] auto charmap_t<T, launch_t::cpu, mode_t::ansi>::getCell(const T& color, F distance) const noexcept -> std::string
{
	const auto colors = similar2(color, distance);

	// Calculate character index
	const int char_pos = colors.fg_delta == 0 ?
		m_nchars - 1 :
		colors.bg_delta * (m_nchars - 1) / colors.fg_delta;

	// Calculate cell position in charmap
	const auto cell_x = char_pos;
	const auto cell_y = (colors.bg_index * m_ncolors + colors.fg_index);

	return m_ansi_colors[cell_y * m_nchars + cell_x];
}

template <typename T>
template <typename D>
[[nodiscard]] auto charmap_t<T, launch_t::cpu, mode_t::ansi>::similar2(const T& goal, D(*distance)(const T&, const T&)) const noexcept -> SimilarColors<D>
{
	const auto start_color = m_colormap.begin<T>();
	auto delta1 = distance(goal, *start_color);
	auto delta2 = delta1;
	auto color1 = start_color;
	auto color2 = color1;

	for (auto color = start_color + 1; color != m_colormap.end<T>(); ++color)
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
