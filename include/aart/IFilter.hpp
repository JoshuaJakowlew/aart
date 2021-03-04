#ifndef AART_I_FILTER_H
#define AART_I_FILTER_H

template <typename T>
concept Filter = 
    requires(T&& t, typename T::resource_t& resource) { { t.operator ()(resource) } -> std::convertible_to<typename T::resource_t&>; }
 ;

template <typename T, typename Resource>
struct IFilter
{
    using resource_t = Resource;

    IFilter() { static_assert(Filter<T>, "IFilter not implemented properly"); };
    IFilter(IFilter&&) = default;
};

#endif
