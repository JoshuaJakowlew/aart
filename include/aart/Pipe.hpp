#ifndef AART_PIPE_H
#define AART_PIPE_H

#include <aart/IFilter.hpp>

namespace detail {
    template <typename, typename>
    struct PipeNode {};

    template<Filter T, Filter U>
    struct PipeNode<T, U> {
        using input_t = typename T::resource_t;
        using output_t = decltype(std::declval<U>()(std::declval<T>()(std::declval<input_t&>())));

        T left;
        U right;

        decltype(auto) operator()(input_t& frame) const
        {
            return right(left(frame));
        }
    };

    template <typename Pipe, Filter T>
        requires std::is_convertible_v<typename Pipe::output_t, typename T::resource_t>
    struct PipeNode<Pipe, T> {
        using input_t = typename Pipe::input_t;
        using output_t = decltype(std::declval<T>()(std::declval<Pipe>()(std::declval<input_t&>())));

        Pipe left;
        T right;

        decltype(auto) operator()(input_t& frame) const
        {
            return right(left(frame));
        };
    };
} // namespace detail

template <Filter T, Filter U>
auto addToPipe(T&& left, U&& right)
{
    return detail::PipeNode<T, U>{std::move(left), std::move(right)};
}

template <typename Pipe, Filter U>
auto addToPipe(Pipe&& left, U&& right)
{
    return detail::PipeNode<Pipe, U>{std::move(left), std::move(right)};
}

template <Filter T, Filter U>
auto operator | (T&& left, U&& right)
{
    return addToPipe(std::move(left), std::move(right));
}

template <typename Pipe, Filter U>
auto operator | (Pipe&& left, U&& right)
{
    return addToPipe(std::move(left), std::move(right));
}

template <typename Pipe, typename Resource>
auto operator |= (Resource&& resource, Pipe&& pipe)
{
    return pipe(resource);
}

#endif
