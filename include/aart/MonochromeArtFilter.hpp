#ifndef AART_MONCHROME_ART_FILTER_H
#define AART_MONCHROME_ART_FILTER_H

#include <opencv2/opencv.hpp>

#include <aart/utility.hpp>
#include <aart/IFilter.hpp>
#include <aart/Charmap.hpp>

class MonochromeArtFilter final : public IFilter<MonochromeArtFilter, cv::Mat, cv::Mat>
{
public:
    MonochromeArtFilter(Charmap& charmap) :
        m_charmap{charmap}
    {}
    MonochromeArtFilter(MonochromeArtFilter&&) = default;

    [[nodiscard]] auto operator ()(input_t&& frame) const -> output_t
    {
        const auto art_w = Width{frame.cols * m_charmap.cellW()};
        const auto art_h = Height{frame.rows * m_charmap.cellH()};
        output_t art{art_w, art_h, m_charmap.type()};

        using pixel_t = unsigned char;
        frame.forEach<pixel_t>([this, &art](pixel_t& p, const int pos[]){
            const auto y = PosY{pos[0]};
            const auto x = PosX{pos[1]};

            const auto cell_w = Width{this->m_charmap.cellW()};
            const auto cell_h = Height{this->m_charmap.cellH()};
            const auto roi = RegionOfInterest{{
                PosX{x * cell_w},
                PosY{y * cell_h},
                Width{cell_w},
                Height{cell_h}
            }};

            const auto nchars = Length{static_cast<int>(this->m_charmap.chars()->size())};
            const int char_idx = static_cast<int>(std::roundf(p * (nchars - 1) / 255.f));
            auto cell = Submatrix<float, 3>{this->m_charmap.cellAt(PosX{char_idx}, PosY{1})};
            cell->copyTo(art(roi));
        });

        return art;
    }
private:
    Charmap& m_charmap;
};

#endif
