#ifndef AART_UTILITY_H
#define AART_UTILITY_H

#include <concepts>

#include <NamedType/named_type.hpp>

using Filename = fluent::NamedType<std::string, struct FilenameTag, fluent::Callable, fluent::Printable>;
using FilenameView = fluent::NamedType<std::string_view, struct FilenameViewTag, fluent::Callable, fluent::Printable>;
template <std::floating_point T>
using Scale = fluent::NamedType<T, struct ScaleTag, fluent::Callable, fluent::Printable>;

#endif
