#ifndef KERNELS_CUDA_H
#define KERNELS_CUDA_H

#include "Colors.h"
#include "Charmap.h"

//namespace cuda {
//	extern void run_kernel(const cv::cuda::GpuMat& pic, cv::cuda::GpuMat& art, const Charmap<lab_t<float>>& charmap);
//}

namespace cuda
{
	template <typename T>
	[[nodiscard]] auto create_art(cv::cuda::GpuMat& pic, const Charmap<T>& charmap) -> cv::cuda::GpuMat
	{
		const auto cellw = charmap.cellW();
		const auto cellh = charmap.cellH();

		cv::cuda::resize(pic, pic, {}, 1.0, (double)cellw / cellh, cv::INTER_LINEAR);
		pic = convertTo<T>(pic);

		const auto picw = pic.size().width;
		const auto pich = pic.size().height;

		auto art = cv::cuda::GpuMat(pich * cellh, picw * cellw, pic.type());

		auto roi = cv::Rect(0, 0, picw, pich);
		auto artroi = art(roi);
		pic.copyTo(artroi);

		//run_kernel(pic, art, charmap);

		/*pic.forEach<T>([&art, &charmap](auto p, const int* pos) noexcept {
			const auto y = pos[0];
			const auto x = pos[1];

			const auto cellw = charmap.cellW();
			const auto cellh = charmap.cellH();

			auto cell = charmap.getCell(p, RGB_euclidian_sqr);
			const auto roi = cv::Rect{ x * cellw, y * cellh, cellw, cellh };
			cell.copyTo(art(roi));
		});*/

		art = convertTo<rgb_t<uint8_t>>(art);
		return art;
	}
}

#endif
