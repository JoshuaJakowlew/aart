#ifndef COMPARATORS_H
#define COMPARATORS_H

#include "colors.h"

[[nodiscard]] inline auto CIE76_distance_sqr(const lab_t<float>& x, const lab_t<float>& y) noexcept -> float
{
	return powf(x.l - y.l, 2) + powf(x.a - y.a, 2) + powf(x.b - y.b, 2);
}

[[nodiscard]] inline auto CIE76_distance(const lab_t<float>& x, const lab_t<float>& y) noexcept -> float
{
	return sqrtf(CIE76_distance_sqr(x, y));
}

[[nodiscard]] inline auto CIE94_distance_sqr(const lab_t<float>& x, const lab_t<float>& y) noexcept -> float
{
	const auto dL = x.l - y.l;

	const auto C1 = sqrtf(x.a * x.a + x.b * x.b);
	const auto C2 = sqrtf(y.a * y.a + y.b * y.b);
	const auto dC = C1 - C2;

	const auto da = x.a - y.a;
	const auto db = x.b - y.b;

	auto dH = da * da + db * db - dC * dC;
	dH = dH > 0.f ? sqrtf(dH) : 0.f;

	// For graphic arts (acc. to Wikipedia)
	constexpr auto kL = 1.f;
	constexpr auto K1 = 0.045f;
	constexpr auto K2 = 0.015f;

	constexpr auto SL = 1.f;
	const auto SC = 1.f + K1 * C1;
	const auto SH = 1.f + K2 * C1;

	// Assume that kC and kH are both unity (acc. to Wikipedia)
	const auto dE = powf(dL / (kL * SL), 2) + powf(dC / SC, 2) + powf(dH / SH, 2);
	return dE;
}

[[nodiscard]] inline auto CIE94_distance(const lab_t<float>& x, const lab_t<float>& y) noexcept -> float
{
	return sqrtf(CIE94_distance_sqr(x, y));
}

[[nodiscard]] inline auto RGB_euclidian_sqr(const rgb_t<float>& x, const rgb_t<float>& y) noexcept -> float
{
	return pow(x.r - y.r, 2) + pow(x.g - y.g, 2) + pow(x.b - y.b, 2);
}

#endif
