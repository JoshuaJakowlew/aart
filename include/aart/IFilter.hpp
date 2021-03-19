#ifndef AART_I_FILTER_H
#define AART_I_FILTER_H

template <typename T>
concept Filter = 
    requires(T&& t, typename T::input_t&& resource) { { t.operator ()(std::move(resource)) } -> std::convertible_to<typename T::output_t>; }
 ;

template <typename T, typename InputResource, typename OutputResource>
struct IFilter
{
    using input_t = InputResource;
    using output_t = OutputResource;

    IFilter() { static_assert(Filter<T>, "IFilter not implemented properly"); };
    IFilter(IFilter&&) = default;
};

#endif
