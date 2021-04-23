#ifndef AART_UTILITY_H
#define AART_UTILITY_H

#include <concepts>

#include <NamedType/named_type.hpp>
#include <opencv2/opencv.hpp>

using Filename = fluent::NamedType<std::string, struct FilenameTag, fluent::Callable, fluent::Printable>;
using FilenameView = fluent::NamedType<std::string_view, struct FilenameViewTag, fluent::Callable, fluent::Printable>;

template <typename T>
struct LengthTag {};

template <std::integral T>
using Length_ = fluent::NamedType<T, LengthTag<T>, fluent::Callable, fluent::Printable>;
using Length = Length_<int>;

template <typename T>
struct IndexTag {};

template <std::integral T>
using Index_ = fluent::NamedType<T, IndexTag<T>, fluent::Callable, fluent::Printable>;
using Index = Index_<int>;

template <typename T>
struct ScaleXTag {};

template <typename T>
    requires std::integral<T> || std::floating_point<T>
using ScaleX_ = fluent::NamedType<T, ScaleXTag<T>, fluent::Callable, fluent::Printable>;
using ScaleX = ScaleX_<double>;

template <typename T>
struct ScaleYTag {};

template <typename T>
    requires std::integral<T> || std::floating_point<T>
using ScaleY_ = fluent::NamedType<T, ScaleYTag<T>, fluent::Callable, fluent::Printable>;
using ScaleY = ScaleY_<double>;

template <typename T>
struct WidthTag {};

template <typename T>
    requires std::integral<T> || std::floating_point<T>
using Width_ = fluent::NamedType<T, WidthTag<T>, fluent::Callable, fluent::Printable>;
using Width = Width_<int>;

template <typename T>
struct HeightTag {};

template <typename T>
    requires std::integral<T> || std::floating_point<T>
using Height_ = fluent::NamedType<T, HeightTag<T>, fluent::Callable, fluent::Printable>;
using Height = Height_<int>;

template <typename T>
struct PosXTag {};

template <typename T>
    requires std::integral<T> || std::floating_point<T>
using PosX_ = fluent::NamedType<T, PosXTag<T>, fluent::Callable, fluent::Printable>;
using PosX = PosX_<int>;

template <typename T>
struct PosYTag {};
template <typename T>
    requires std::integral<T> || std::floating_point<T>
using PosY_ = fluent::NamedType<T, PosYTag<T>, fluent::Callable, fluent::Printable>;
using PosY = PosY_<int>;

using RegionOfInterest = fluent::NamedType<cv::Rect, struct RegionOfInterestTag, fluent::Callable, fluent::Printable>;

template <typename T, int Channels>
struct MatrixTag {};

template <typename T, int Channels>
using Matrix = fluent::NamedType<cv::Mat, MatrixTag<T, Channels>, fluent::Callable, fluent::Printable>;

template <typename T, int Channels>
struct SubmatrixTag {};
template <typename T, int Channels>
using Submatrix = fluent::NamedType<cv::Mat, struct SubmatrixTag<T, Channels>, fluent::Callable, fluent::Printable>;

template <typename T>
struct MultiplyTag {};

template <typename T>
using Multiply_ = fluent::NamedType<T, MultiplyTag<T>, fluent::Callable, fluent::Printable>;
using Multiply = Multiply_<double>;
template <typename T>
struct AddTag {};

template <typename T>
using Add_ = fluent::NamedType<T, AddTag<T>, fluent::Callable, fluent::Printable>;
using Add = Add_<double>;


template <typename T>
struct DistanceTag {};

template <typename T>
using Distance_ = fluent::NamedType<T, DistanceTag<T>, fluent::Callable, fluent::Printable>;
using Distance = Distance_<double>;


#endif
