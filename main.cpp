#include <iostream>
#include <type_traits>
#include <concepts>
#include <vector>
#include <variant>
#include <functional>
#include <tuple>

#include <opencv2/opencv.hpp>
#include <ft2build.h>
#include FT_FREETYPE_H

#include <aart/utils.h>
#include <aart/Charmap.hpp>
#include <aart/Pipe.hpp>
#include <aart/ImageManager.hpp>
#include <aart/ScaleFilter.hpp>
#include <aart/GrayscaleFilter.hpp>
#include <aart/MonochromeArtFilter.hpp>

void show(const cv::Mat& x)
{
    cv::namedWindow("Hello!");
    cv::imshow("Hello!", x);
    cv::waitKey(0);
    cv::destroyWindow("Hello!");
}

int main()
{
    Charmap chr{
        "Courier New.ttf",
        14,
        " .:-=+*#%@",
        {{0, 0, 0}, {255, 255, 255}}
    };
    show(chr.render());

    ImageManager imageManager{"test.png"};
    auto img = imageManager.getResource();
    
    auto art = std::move(img) |= ScaleFilter{0.5f, 0.5f}
                              |  GrayscaleFilter{}
                              |  MonochromeArtFilter{chr};
    
    art.convertTo(art, CV_8U, 255); // Shit
    imageManager.assign(std::move(art));
    show(imageManager.getResource());
    imageManager.write("result.png");
}