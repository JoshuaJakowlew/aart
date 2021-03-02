#include <aart/ImageManager.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <type_traits>
#include <aart/utils.h>
#include <concepts>
#include <vector>
#include <variant>

template <ResourceManager T>
void show(T&& x)
{
    cv::namedWindow("Hello!");
    cv::imshow("Hello!", x.getResource());
    cv::waitKey(0);
    cv::destroyWindow("Hello!");
}

int main()
{
    ImageManager imageManager;
    auto img = imageManager.read("C:/Users/jakow/Documents/Programming/aart/test.png");
    
    show(std::move(imageManager));
}