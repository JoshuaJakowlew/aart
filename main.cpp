#include <iostream>
#include <opencv2/opencv.hpp>
#include <type_traits>
#include <concepts>
#include <vector>
#include <variant>
#include <functional>
#include <tuple>

#include <aart/utils.h>
#include <aart/ImageManager.hpp>
#include <aart/ScaleFilter.hpp>
#include <aart/GrayscaleFilter.hpp>

void show(const cv::Mat& x)
{
    cv::namedWindow("Hello!");
    cv::imshow("Hello!", x);
    cv::waitKey(0);
    cv::destroyWindow("Hello!");
}

template<typename, typename>
struct PipeNode {};

template<Filter T, Filter U>
struct PipeNode<T, U> {
    using input_t = typename T::resource_t;
    using return_t = decltype(std::declval<U>()(std::declval<T>()(std::declval<input_t&>())));

    T left;
    U right;

    decltype(auto) operator()(input_t& frame) const
    {
        return right(left(frame));
    }
};

template<typename Pipe, Filter T>
    requires std::is_convertible_v<typename Pipe::return_t, typename T::resource_t>
struct PipeNode<Pipe, T> {
    using input_t = typename Pipe::input_t;
    using return_t = decltype(std::declval<T>()(std::declval<Pipe>()(std::declval<input_t&>())));

    Pipe left;
    T right;

    decltype(auto) operator()(input_t& frame) const
    {
        return right(left(frame));
    };
};

template<Filter T, Filter U>
auto addToPipe(T&& left, U&& right) {
    return PipeNode<T, U>{std::move(left), std::move(right)};
}

template<typename Pipe, Filter U>
auto addToPipe(Pipe&& left, U&& right) {
    return PipeNode<Pipe, U>{std::move(left), std::move(right)};
}

int main()
{
    ImageManager imageManager;
    auto img = imageManager.read("C:/Users/jakow/Documents/Programming/aart/test.png");
    show(img);

    //ScaleFilter scaleFilter{0.5f, 0.5f};
    //scaleFilter(img);

    //GrayscaleFilter grayscaleFilter{};
    //grayscaleFilter(img);

    //auto pipe = addToPipe(ScaleFilter{0.5f, 0.5f}, GrayscaleFilter{});
    //auto pipe2 = addToPipe(std::move(pipe), ScaleFilter{2.f, 2.f});
    //auto pipe3 = addToPipe(std::move(pipe2), ScaleFilter{2.f, 2.f});
    auto pipe = addToPipe(addToPipe(addToPipe(ScaleFilter{0.5f, 0.5f}, GrayscaleFilter{}), ScaleFilter{2.f, 2.f}), ScaleFilter{2.f, 2.f});
    pipe(img);
    show(img);

}