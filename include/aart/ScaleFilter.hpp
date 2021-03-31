#ifndef AART_SCALE_FILTER_H
#define AART_SCALE_FILTER_H

#include <opencv2/opencv.hpp>

#include <aart/IFilter.hpp>
#include <aart/utility.hpp>

template <typename T, int Channels>
class ScaleFilter final : public IFilter<ScaleFilter<T, Channels>, Matrix<T, Channels>>
{
public:
    using typename ScaleFilter<T, Channels>::input_t;
    using typename ScaleFilter<T, Channels>::output_t;

    ScaleFilter(ScaleX scaleX, ScaleY scaleY) :
        m_scaleX{scaleX},
        m_scaleY{scaleY}
    {}
    ScaleFilter(ScaleFilter&&) = default;

    [[nodiscard]] auto operator ()(input_t&& frame) const -> output_t
    {
        output_t result;
        const auto interpolation_method = (m_scaleX < 1. && m_scaleY < 1.) ? cv::INTER_AREA : cv::INTER_CUBIC;
        cv::resize(
            frame.get(),
            result.get(),
            cv::Size{}, // Empty size means "convert with scale, not exact size"
            m_scaleX,
            m_scaleY,
            interpolation_method
        );
        return result;
    }
private:
    ScaleX m_scaleX{1.};
    ScaleY m_scaleY{1.};
};

#endif
