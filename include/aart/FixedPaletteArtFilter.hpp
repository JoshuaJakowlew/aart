#ifndef AART_FIXED_PALETTE_FILTER
#define AART_FIXED_PALETTE_FILTER

#include <opencv2/opencv.hpp>

#include <aart/utility.hpp>
#include <aart/IFilter.hpp>
#include <aart/Charmap.hpp>
#include <aart/Metrics.hpp>


template <MetricType MetricT>
class FixedPaletteFilter final : public IFilter<
    FixedPaletteFilter<MetricT>,
    Matrix<unsigned char, 3>,
    Matrix<float, 3>>
{
public:
    using typename FixedPaletteFilter<MetricT>::input_t;
    using typename FixedPaletteFilter<MetricT>::output_t;

    FixedPaletteFilter(Charmap& charmap) :
        m_charmap{charmap}
    {}
    FixedPaletteFilter(FixedPaletteFilter&&) = default;

    [[nodiscard]] auto operator ()(input_t&& frame) const -> output_t
    {
        const auto art_w = Width{frame->cols * m_charmap.cellW()};
        const auto art_h = Height{frame->rows * m_charmap.cellH()};
        output_t art{{art_w, art_h, m_charmap.type()}};

        frame->forEach<pixel_t>([this, &art](pixel_t& p, const int pos[]){
            const auto y = PosY{pos[0]};
            const auto x = PosX{pos[1]};

            const auto cell_w = this->m_charmap.cellW();
            const auto cell_h = this->m_charmap.cellH();
            const auto roi = RegionOfInterest{{
                PosX{x * cell_w},
                PosY{y * cell_h},
                cell_w,
                cell_h
            }};

            const auto [bg_delta, fg_delta, bg_index, fg_index] = similar2(p);

            const auto nchars = m_charmap.nChars();
			// Calculate character index
			const auto cell_x = PosX{fg_delta == 0 ?
				static_cast<int>(m_charmap.nChars() - 1) :
				static_cast<int>(bg_delta * (m_charmap.nChars() - 1) / fg_delta)};

			// Calculate cell position in charmap
			const auto cell_y = PosY{static_cast<int>(bg_index * this->m_charmap.colors()->size() + fg_index)};

            auto cell = output_t{this->m_charmap.cellAt(cell_x, cell_y)};
            cell->copyTo(art.get()(roi));
        });

        return art;
    }
private:
    using pixel_t = cv::Point3_<unsigned char>;
    Charmap& m_charmap;

    [[nodiscard]] auto similar2(pixel_t const & target) const noexcept -> std::tuple<Distance, Distance, Index, Index>
    {
        auto colors = m_charmap.colors();
        const auto start_color = colors->cbegin();
        
		auto delta1 = Metric<MetricT>::template distance<unsigned char>(target, *start_color);
		auto delta2 = delta1;
		auto color1 = start_color;
		auto color2 = color1;
		for (auto color = start_color + 1; color != colors->cend(); ++color)
		{
			const auto delta = Metric<MetricT>::template distance<unsigned char>(target, *color);
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
		const int index1 = static_cast<int>(color1 - start_color);
		const int index2 = static_cast<int>(color2 - start_color);
		return {
			Distance{delta1}, Distance{delta2},
			Index{index1}, Index{index2}
		};
    }
};

#endif
