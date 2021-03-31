#ifndef AART_CONVERT_FILTER_H
#define AART_CONVERT_FILTER_H

#include <limits>

#include <opencv2/opencv.hpp>

#include <aart/IFilter.hpp>
#include <aart/utility.hpp>

template <typename InputT, int InputChannels, typename OutputT, int OutputChannels>
    requires
        (InputChannels > 0) &&
        (InputChannels <= 4) &&
        (OutputChannels > 0) &&
        (OutputChannels <= 4) &&
        std::convertible_to<InputT, OutputT> &&
        (std::integral<InputT> || std::floating_point<InputT>) &&
        (std::integral<OutputT> || std::floating_point<OutputT>) &&
        (!std::same_as<InputT, unsigned>) &&
        (!std::same_as<OutputT, unsigned>) &&
        (!std::same_as<InputT, long long>) &&
        (!std::same_as<OutputT, long long>) &&
        (!std::same_as<InputT, unsigned long long>) &&
        (!std::same_as<OutputT, unsigned long long>) &&
        (!std::same_as<InputT, long double>) &&
        (!std::same_as<OutputT, long double>)
class ConvertFilter final : public IFilter<
    ConvertFilter<InputT, InputChannels, OutputT, OutputChannels>,
    Matrix<InputT, InputChannels>,
    Matrix<OutputT, OutputChannels>>
{
public:
    using typename ConvertFilter<InputT, InputChannels, OutputT, OutputChannels>::input_t;
    using typename ConvertFilter<InputT, InputChannels, OutputT, OutputChannels>::output_t;

    ConvertFilter(Multiply alpha = Multiply{1.}, Add beta = Add{0.}):
        m_alpha{alpha},
        m_beta{beta}
    {}
    ConvertFilter(ConvertFilter&&) = default;

    [[nodiscard]] auto operator ()(input_t&& frame) const -> output_t
        requires
            std::convertible_to<input_t, cv::Mat> &&
            std::convertible_to<output_t, cv::Mat>
    {
        output_t result;
        frame->convertTo(result.get(), getOpenCvMatType(), m_alpha, m_beta);
        return result;
    }
private:
    Multiply m_alpha;
    Add m_beta;

    [[nodiscard]] constexpr auto getOpenCvMatType() const noexcept
    {
        if constexpr (std::same_as<OutputT, char>)
            return std::numeric_limits<char>::is_signed ? CV_8SC(OutputChannels) : CV_8UC(OutputChannels);
        else if constexpr (std::same_as<OutputT, unsigned char>)
            return CV_8UC(OutputChannels);
        else if constexpr (std::same_as<OutputT, signed char>)
            return CV_8SC(OutputChannels);
        else if constexpr (std::same_as<OutputT, unsigned short>)
            return CV_16UC(OutputChannels);
        else if constexpr (std::same_as<OutputT, short>)
            return CV_16SC(OutputChannels);
        else if constexpr (std::same_as<OutputT, int>)
            return CV_32SC(OutputChannels);
        else if constexpr (std::same_as<OutputT, float>)
            return CV_32FC(OutputChannels);
        else if constexpr (std::same_as<OutputT, double>)
            return CV_64FC(OutputChannels);
        else
            static_assert(false, "Unsupported type");
    }
};

#endif
