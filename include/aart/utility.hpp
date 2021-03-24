#ifndef AART_UTILITY_H
#define AART_UTILITY_H

#include <concepts>

#include <type_safe/strong_typedef.hpp>
#include <NamedType/named_type.hpp>

template <std::floating_point T>
using scale = fluent::NamedType<T, struct scale_tag, fluent::Callable, fluent::Printable>;

#endif
