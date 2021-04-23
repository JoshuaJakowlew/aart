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

#include <aart/Charmap.hpp>
#include <aart/Pipe.hpp>
#include <aart/Image.hpp>
#include <aart/ScaleFilter.hpp>
#include <aart/ColorSpaceFilter.hpp>
#include <aart/MonochromeArtFilter.hpp>
#include <aart/ConvertFilter.hpp>
#include <aart/FixedPaletteArtFilter.hpp>
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
        FilenameView{"Courier New.ttf"},
        FontSize{14},
        CharPalette{" .:-=+*#%@"},
        ColorPalette{{
            {0, 0, 0}, {64, 64, 64}, {128, 128, 128}, {160, 160, 160}, {192, 192, 192}, {255, 255, 255}
        }}
    };
    chr.render();

    Image atlas = chr.charmap();
    show(atlas.get());
    atlas.write(Filename{"chr.png"});

    const auto fixed_palette_pipe = ScaleFilter<unsigned char, 3>{ScaleX{0.5}, ScaleY{0.5 * chr.ratio()}}
                                  |  ColorSpaceFilter<unsigned char, 3, unsigned char, 3>{cv::COLOR_BGR2Lab}
                                  |  FixedPaletteFilter<MetricType::EuclidianSqr>{chr}
                                  |  ConvertFilter<float, 3, unsigned char, 3>{Multiply{255.}};
                                   
    Image<unsigned char, 3> img{Filename{"test.png"}};
    img = std::move(img) |= fixed_palette_pipe;
    show(img.get());
    img.write(Filename{"result.png"});
}