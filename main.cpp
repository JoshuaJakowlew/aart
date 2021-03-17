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

void show(const cv::Mat& x)
{
    cv::namedWindow("Hello!");
    cv::imshow("Hello!", x);
    cv::waitKey(0);
    cv::destroyWindow("Hello!");
}

int main()
{
    // ImageManager imageManager{"test.png"};
    // show(
    //     imageManager.getResource() |= ScaleFilter{0.5f, 0.5f}
    //                                |  GrayscaleFilter{}
    //                                |  ScaleFilter{2.f, 2.f}
    //                                |  ScaleFilter{2.f, 2.f}
    // );
    // imageManager.write("result.png");

    Charmap chr{"Courier New.ttf", 72, "j .:-=+*#%@gGj", {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}}};
    ImageManager chrm;
    chrm.assign(chr.render());
    chrm.write("font.png");

    auto text = chrm.getResource();
    show(text);

    show(chr.cellAt(0, 5));
}