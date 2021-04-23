#ifndef AART_COLOR_SPACE_FILTER_H
#define AART_COLOR_SPACE_FILTER_H

#include <opencv2/opencv.hpp>

#include <aart/IFilter.hpp>

template <typename InputT, int InputChannels, typename OutputT, int OutputChannels>
class ColorSpaceFilter final : public IFilter<
    ColorSpaceFilter<InputT, InputChannels, OutputT, OutputChannels>,
    Matrix<InputT, InputChannels>,
    Matrix<OutputT, OutputChannels>>
{
public:
    using typename ColorSpaceFilter<InputT, InputChannels, OutputT, OutputChannels>::input_t;
    using typename ColorSpaceFilter<InputT, InputChannels, OutputT, OutputChannels>::output_t;

    ColorSpaceFilter(cv::ColorConversionCodes conversionCode) :
        m_conversionCode{ conversionCode }
    {}
    ColorSpaceFilter(ColorSpaceFilter&&) = default;

    [[nodiscard]] auto operator ()(input_t&& frame) const -> output_t
    {
        output_t result;
        cv::cvtColor(frame.get(), result.get(), m_conversionCode);
        return result;
    }
private:
    cv::ColorConversionCodes m_conversionCode;
};

#endif
