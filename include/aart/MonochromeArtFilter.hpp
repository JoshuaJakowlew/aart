#ifndef AART_MONCHROME_ART_FILTER_H
#define AART_MONCHROME_ART_FILTER_H

#include <opencv2/opencv.hpp>

#include <aart/utility.hpp>
#include <aart/IFilter.hpp>
#include <aart/Charmap.hpp>

class MonochromeArtFilter final : public IFilter<MonochromeArtFilter, cv::Mat, cv::Mat>
{
public:
    using scale_t = scale<double>;

    MonochromeArtFilter(Charmap& charmap, scale_t scaleX = scale_t{1.}, scale_t scaleY = scale_t{1.}) :
        m_scaleX{std::move(scaleX)},
        m_scaleY{std::move(scaleY)},
        m_charmap{charmap}
    {}
    MonochromeArtFilter(MonochromeArtFilter&&) = default;

    [[nodiscard]] auto operator ()(input_t&& frame) const -> output_t
    {
        const auto cell_w = m_charmap.cellW();
        const auto cell_h = m_charmap.cellH();
        const auto horz_scale_ratio = static_cast<double>(cell_w) / cell_h;
        cv::resize(std::forward<input_t>(frame), frame, {}, m_scaleX, m_scaleY * horz_scale_ratio, cv::INTER_AREA);

        auto art_size = frame.size();
        art_size.width *= cell_w;
        art_size.height *= cell_h;
        cv::Mat art{art_size, m_charmap.type()};

        using pixel_t = unsigned char;
        frame.forEach<pixel_t>([this, &art](pixel_t& p, const int pos[]){
            const auto y = pos[0];
            const auto x = pos[1];

            const auto cell_w = this->m_charmap.cellW();
            const auto cell_h = this->m_charmap.cellH();
            const auto roi = cv::Rect{x * cell_w, y * cell_h, cell_w, cell_h};

            const auto char_idx = static_cast<int>(std::roundf(p * (this->m_charmap.chars().length() - 1) / 255.f));
            auto cell = this->m_charmap.cellAt(char_idx, 1);
            cell.copyTo(art(roi));
        });

        return art;
    }
private:
    Charmap& m_charmap;
    scale_t m_scaleX{1.};
    scale_t m_scaleY{1.};
};

#endif
