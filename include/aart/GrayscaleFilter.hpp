#ifndef AART_GRAYSCALE_FILTER_H
#define AART_GRAYSCALE_FILTER_H

#include <opencv2/opencv.hpp>

#include <aart/IFilter.hpp>

template <typename T, int Channels>
class GrayscaleFilter final : public IFilter<
    GrayscaleFilter<T, Channels>,
    Matrix<T, Channels>,
    Matrix<T, 1>>
{
public:
    using typename GrayscaleFilter<T, Channels>::input_t;
    using typename GrayscaleFilter<T, Channels>::output_t;

    GrayscaleFilter() = default;
    GrayscaleFilter(cv::ColorConversionCodes conversionCode) :
        m_conversionCode{ conversionCode }
    {}
    GrayscaleFilter(GrayscaleFilter&&) = default;

    [[nodiscard]] auto operator ()(input_t&& frame) const -> output_t
    {
        output_t result;
        cv::cvtColor(frame.get(), result.get(), m_conversionCode);
        return result;
    }
private:
    cv::ColorConversionCodes m_conversionCode = cv::COLOR_BGR2GRAY;
};

#endif
