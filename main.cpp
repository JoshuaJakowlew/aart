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
#include <aart/Image.hpp>
#include <aart/ScaleFilter.hpp>
#include <aart/GrayscaleFilter.hpp>
#include <aart/MonochromeArtFilter.hpp>
#include <aart/utility.hpp>

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
        "Courier New.ttf", 14, " .:-=+*#%@", {{0, 0, 0}, {255, 255, 255}}
    };
    Image chrm{chr.render()};
    show(chrm.get());
    chrm.write("chr.png");

    Image img{"test.png"};
    auto art = std::move(img.get()) |= GrayscaleFilter{}
                                    |  MonochromeArtFilter{chr, scale{0.5}, scale{0.5}};
    
    art.convertTo(art, CV_8U, 255); // Shit
    img.assign(std::move(art));
    
    show(img.get());
    img.write("result.png");
}