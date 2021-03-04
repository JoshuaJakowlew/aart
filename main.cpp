#include <iostream>
#include <opencv2/opencv.hpp>
#include <type_traits>
#include <concepts>
#include <vector>
#include <variant>
#include <functional>
#include <tuple>

#include <aart/utils.h>
#include <aart/Pipe.hpp>
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

int main()
{
    ImageManager imageManager{"C:/Users/jakow/Documents/Programming/aart/test.png"};
    show(
        imageManager.getResource() |= ScaleFilter{0.5f, 0.5f}
                                   |  GrayscaleFilter{}
                                   |  ScaleFilter{2.f, 2.f}
                                   |  ScaleFilter{2.f, 2.f}
    );
}