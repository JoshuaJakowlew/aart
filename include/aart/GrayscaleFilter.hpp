#ifndef AART_GRAYSCALE_FILTER_H
#define AART_GRAYSCALE_FILTER_H

#include <opencv2/opencv.hpp>

#include <aart/IFilter.hpp>

class GrayscaleFilter final : public IFilter<GrayscaleFilter, cv::Mat, cv::Mat>
{
public:
    GrayscaleFilter() = default;
    GrayscaleFilter(cv::ColorConversionCodes conversionCode) :
        m_conversionCode{ conversionCode }
    {}
    GrayscaleFilter(GrayscaleFilter&&) = default;

    [[nodiscard]] auto operator ()(input_t&& frame) const -> output_t
    {
        cv::Mat result;
        cv::cvtColor(std::forward<input_t>(frame), result, m_conversionCode);
        return result;
    }
private:
    cv::ColorConversionCodes m_conversionCode = cv::COLOR_BGR2GRAY;
};

#endif
