#ifndef AART_PIPE_H
#define AART_PIPE_H

#include <aart/IFilter.hpp>

namespace detail {
    template <typename, typename>
    struct PipeNode {};

    template<Filter T, Filter U>
    struct PipeNode<T, U> {
        using input_t = typename T::input_t;
        using output_t = decltype(std::declval<U>()(
            std::declval<T>()(
                std::declval<input_t&&>()
            )
        ));

        T left;
        U right;

        output_t operator()(input_t&& frame) const
        {
            return right(left(std::forward<input_t>(frame)));
        }
    };

    template <typename Pipe, Filter T>
        requires std::is_convertible_v<typename Pipe::output_t, typename T::resource_t>
    struct PipeNode<Pipe, T> {
        using input_t = typename Pipe::input_t;
        using output_t = decltype(std::declval<T>()(
            std::declval<Pipe>()(
                std::declval<input_t&&>()
            )
        ));

        Pipe left;
        T right;

        output_t operator()(input_t&& frame) const
        {
            return right(left(std::forward<input_t>(frame)));
        };
    };
} // namespace detail

template <Filter T, Filter U>
auto addToPipe(T&& left, U&& right)
{
    return detail::PipeNode<T, U>{std::forward<T>(left), std::forward<U>(right)};
}

template <typename Pipe, Filter U>
auto addToPipe(Pipe&& left, U&& right)
{
    return detail::PipeNode<Pipe, U>{std::forward<Pipe>(left), std::forward<U>(right)};
}

template <Filter T, Filter U>
auto operator | (T&& left, U&& right)
{
    return addToPipe(std::forward<T>(left), std::forward<U>(right));
}

template <typename Pipe, Filter U>
auto operator | (Pipe&& left, U&& right)
{
    return addToPipe(std::forward<Pipe>(left), std::forward<U>(right));
}

template <typename Pipe, typename Resource>
auto operator |= (Resource&& resource, Pipe&& pipe)
{
    return pipe(std::forward<Resource>(resource));
}

#endif
