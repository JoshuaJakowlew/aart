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
    Charmap chr{ FilenameView{"Courier New.ttf"}
               , FontSize{14}
               , CharPalette{" .:-=+*#%@"}
               , ColorPalette{{{0, 0, 0}, {255, 255, 255}}}
               };
    chr.render();

    Image atlas(std::move(chr.charmap().get()));
    show(atlas.get());
    atlas.write(Filename{"chr.png"});

    Image img{Filename{"test.png"}};
    Image art = std::move(img.get()) |= ScaleFilter{ScaleX{0.5}, ScaleY{0.5 * chr.ratio()}}
                                     |  GrayscaleFilter{}
                                     |  MonochromeArtFilter{chr};
    art.get().convertTo(art.get(), CV_8U, 255); // Shit, move to filter
    show(art.get());
    art.write(Filename{"result.png"});
}