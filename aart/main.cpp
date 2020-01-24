#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

int main()
{
	using Pixel = cv::Point3_<float>;

    auto palette1 = cv::imread("palette.bmp", cv::IMREAD_COLOR);
	cv::Mat palette;
	palette1.convertTo(palette, CV_32F);
	palette *= 1.0 / 255.0;
    cv::cvtColor(palette, palette, cv::COLOR_BGR2Lab);

	auto pic1 = cv::imread("test.jpg", cv::IMREAD_COLOR);
	cv::Mat pic;
	pic1.convertTo(pic, CV_32F);
	pic *= 1.0 / 255.0;
	auto orig = pic.clone();
	auto test = *pic.begin<Pixel>();
	cv::cvtColor(pic, pic, cv::COLOR_BGR2Lab);
	auto test1 = *pic.begin<Pixel>();

	auto dist = [](Pixel y, Pixel x) {
		return sqrt(pow(x.x - y.x, 2) + pow(x.y - y.y, 2) + pow(x.z - y.z, 2));
	};

	auto similar = [&palette, dist](const Pixel& goal) {
		return std::accumulate(palette.begin<Pixel>() + 1, palette.end<Pixel>(), *palette.begin<Pixel>(), [&goal, dist](auto prev, auto curr) {
			auto currD = dist(goal, curr);
			auto prevD = dist(goal, prev);
			return (currD < prevD ? curr : prev);
		});
	};

	pic.forEach<Pixel>([&palette, similar](Pixel& p, const int* pos) {
		p = similar(p);
	});

	cv::cvtColor(pic, pic, cv::COLOR_Lab2BGR);

	cv::imshow("1", orig);
	cv::imshow("2", pic);
	cv::waitKey(0);
}
