#ifndef CHARMAP_H
#define CHARMAP_H

#include "Colors.h"

template <typename T, typename D, typename I = int>
struct SimilarColors
{
	T bg;
	T fg;
	D bg_delta;
	D fg_delta;
	I bg_index;
	I fg_index;
};

template <typename T>
class Charmap
{
public:
	Charmap(cv::Mat charmap, cv::Mat colormap, std::string chars) :
		m_charmap{ std::move(charmap) },
		m_colormap{ convertTo<T>(std::move(colormap)) },
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
	[[nodiscard]] auto similar2(const T& goal, D(*distance)(const T&, const T&)) const noexcept -> SimilarColors<T, D>;
};

template <typename T>
template <typename F>
[[nodiscard]] auto Charmap<T>::getCell(const T& color, F distance) const noexcept -> cv::Mat
{
	const auto colors = similar2(color, distance);

	// Calculate character index
	const int char_pos = colors.fg_delta == 0 ?
		m_nchars - 1 :
		colors.bg_delta / colors.fg_delta * (m_nchars - 1);

	// Calculate cell position in charmap
	const auto cell_x = char_pos * m_cellw;
	const auto cell_y = (colors.bg_index * m_ncolors + colors.fg_index) * m_cellh;

	return m_charmap(cv::Rect{ cell_x, cell_y, m_cellw, m_cellh });
}

template <typename T>
template <typename D>
[[nodiscard]] auto Charmap<T>::similar2(const T& goal, D(*distance)(const T&, const T&)) const noexcept -> SimilarColors<T, D>
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
		*color1, *color2,
		 delta1,  delta2,
		 index1,  index2
	};
}

#endif
