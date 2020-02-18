#ifndef COMPARATORS_H
#define COMPARATORS_H

#include "colors.h"

[[nodiscard]] inline auto CIE76_distance(const lab_t<float>& x, const lab_t<float>& y) noexcept -> float
{
	return sqrt(pow(x.l - y.l, 2) + pow(x.a - y.a, 2) + pow(x.b - y.b, 2));
}

[[nodiscard]] inline auto CIE76_distance_sqr(const lab_t<float>& x, const lab_t<float>& y) noexcept -> float
{
	return pow(x.l - y.l, 2) + pow(x.a - y.a, 2) + pow(x.b - y.b, 2);
}

[[nodiscard]] inline auto RGB_euclidian_sqr(const rgb_t<float>& x, const rgb_t<float>& y) noexcept -> float
{
	return pow(x.r - y.r, 2) + pow(x.g - y.g, 2) + pow(x.b - y.b, 2);
}

#endif
