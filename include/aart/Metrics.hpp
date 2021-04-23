#ifndef AART_METRICS_H
#define AART_METRICS_H

#include <opencv2/opencv.hpp>

#include <utility>

enum class MetricType
{
    EuclidianSqr
};

template <MetricType MetricT, typename DistanceT = double>
struct Metric
{
    //static_assert(false, "Unspecified metric")
};

template <typename DistanceT>
struct Metric<MetricType::EuclidianSqr, DistanceT>
{
    using distance_t = Distance_<DistanceT>;

    template <typename T>
    [[nodiscard]] constexpr static auto distance(cv::Point3_<T> const & a_, cv::Point3_<T> const & b_) noexcept -> distance_t
    {
        cv::Point3d a = a_;
        cv::Point3d b = b_;
        return distance_t{(a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z)};
    }
};

#endif
