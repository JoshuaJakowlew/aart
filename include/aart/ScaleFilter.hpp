#ifndef AART_SCALE_FILTER_H
#define AART_SCALE_FILTER_H

#include <opencv2/opencv.hpp>

#include <aart/IFilter.hpp>

class ScaleFilter final : public IFilter<ScaleFilter, cv::Mat>
{
public:
    ScaleFilter(float scaleX, float scaleY) :
        m_scaleX{ scaleX },
        m_scaleY{ scaleY }
    {}
    ScaleFilter(ScaleFilter&&) = default;

    [[nodiscard]] auto operator ()(resource_t& frame) const -> resource_t&
    {
        cv::resize(frame, frame, {}, m_scaleX, m_scaleY);
        return frame;
    }
private:
    float m_scaleX = 1.f;
    float m_scaleY = 1.f;
};

#endif
