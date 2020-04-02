#ifndef COLOR_QUANTIZATION_H
#define COLOR_QUANTIZATION_H

#include <opencv2/opencv.hpp>

auto palette(cv::InputArray picture, int colors)->cv::Mat;
auto histogram(cv::InputArray picture, int bins)->std::vector<cv::Mat>;
auto find_dominant_colors(cv::Mat img, int count) ->std::vector<cv::Vec3b>;

#endif // COLOR_QUANTIZATION_H
