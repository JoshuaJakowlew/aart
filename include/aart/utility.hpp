#ifndef AART_UTILITY_H
#define AART_UTILITY_H

#include <concepts>

#include <type_safe/strong_typedef.hpp>

namespace ts = type_safe;

template <std::floating_point T = float>
struct scale :
    ts::strong_typedef<scale<T>, T>,
    ts::strong_typedef_op::floating_point_arithmetic<scale<T>>
{
    using ts::strong_typedef<scale<T>, T>::strong_typedef;
};

template <std::floating_point T>
scale(T) -> scale<T>;

#endif
