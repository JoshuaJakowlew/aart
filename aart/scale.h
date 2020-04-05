#ifndef SCALE_H
#define SCALE_H

#include <opencv2/opencv.hpp>

[[nodiscard]] auto scale_const(const cv::Mat& img, cv::Size size) -> cv::Mat
{
	cv::Mat result;
	cv::resize(img, result, size, 0, 0);
	return result;
}

[[nodiscard]] auto scale_factor(const cv::Mat& img, cv::Size2f factors) -> cv::Mat
{
	cv::Mat result;
	cv::resize(img, result, {}, factors.width, factors.height);
	return result;
}

[[nodiscard]] auto scale_fit(const cv::Mat& img, cv::Size size) -> cv::Mat
{
	constexpr int cellw = 7;
	constexpr int cellh = 11;

	cv::Mat result;
	cv::resize(img, result, { size.width / cellw, size.height / cellh }, 0, 0);
	return result;
}


#endif // SCALE_H
