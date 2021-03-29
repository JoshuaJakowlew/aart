#ifndef AART_UTILITY_H
#define AART_UTILITY_H

#include <concepts>

#include <NamedType/named_type.hpp>
#include <opencv2/opencv.hpp>

using Filename = fluent::NamedType<std::string, struct FilenameTag, fluent::Callable, fluent::Printable>;
using FilenameView = fluent::NamedType<std::string_view, struct FilenameViewTag, fluent::Callable, fluent::Printable>;

template <std::integral T>
using Length_ = fluent::NamedType<T, struct LengthTag, fluent::Callable, fluent::Printable>;
using Length = Length_<int>;
template <std::integral T>
using Index_ = fluent::NamedType<T, struct IndexTag, fluent::Callable, fluent::Printable>;
using Index = Index_<int>;

template <typename T>
    requires std::integral<T> || std::floating_point<T>
using ScaleX_ = fluent::NamedType<T, struct ScaleXTag, fluent::Callable, fluent::Printable>;
using ScaleX = ScaleX_<double>;
template <typename T>
    requires std::integral<T> || std::floating_point<T>
using ScaleY_ = fluent::NamedType<T, struct ScaleYTag, fluent::Callable, fluent::Printable>;
using ScaleY = ScaleY_<double>;

template <typename T>
    requires std::integral<T> || std::floating_point<T>
using Width_ = fluent::NamedType<T, struct WidthTag, fluent::Callable, fluent::Printable>;
using Width = Width_<int>;

template <typename T>
    requires std::integral<T> || std::floating_point<T>
using Height_ = fluent::NamedType<T, struct HeightTag, fluent::Callable, fluent::Printable>;
using Height = Height_<int>;

template <typename T>
    requires std::integral<T> || std::floating_point<T>
using PosX_ = fluent::NamedType<T, struct PosXTag, fluent::Callable, fluent::Printable>;
using PosX = PosX_<int>;

template <typename T>
    requires std::integral<T> || std::floating_point<T>
using PosY_ = fluent::NamedType<T, struct PosYTag, fluent::Callable, fluent::Printable>;
using PosY = PosY_<int>;

using RegionOfInterest = fluent::NamedType<cv::Rect, struct RegionOfInterestTag, fluent::Callable, fluent::Printable>;

template <typename T, int Channels>
using Matrix = fluent::NamedType<cv::Mat, struct NormalizedMatrixTag, fluent::Callable, fluent::Printable>;

template <typename T, int Channels>
using Submatrix = fluent::NamedType<cv::Mat, struct SubmatrixTag, fluent::Callable, fluent::Printable>;
#endif
