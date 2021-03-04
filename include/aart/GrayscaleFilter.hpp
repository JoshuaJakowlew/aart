#ifndef AART_GRAYSCALE_FILTER_H
#define AART_GRAYSCALE_FILTER_H

#include <opencv2/opencv.hpp>

#include <aart/IFilter.hpp>

class GrayscaleFilter final : public IFilter<GrayscaleFilter, cv::Mat>
{
public:
    GrayscaleFilter() = default;
    GrayscaleFilter(cv::ColorConversionCodes conversionCode) :
        m_conversionCode{ conversionCode }
    {}
    GrayscaleFilter(GrayscaleFilter&&) = default;

    [[nodiscard]] auto operator ()(resource_t& frame) const -> resource_t&
    {
        cv::cvtColor(frame, frame, m_conversionCode);
        return frame;
    }
private:
    cv::ColorConversionCodes m_conversionCode = cv::COLOR_BGR2GRAY;
};

#endif
