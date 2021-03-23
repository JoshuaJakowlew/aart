#ifndef AART_I_RESOURCE_H
#define AART_I_RESOURCE_H

#include <concepts>
#include <string>

template <typename T>
concept Resource = 
    requires(T&& t, const std::string& filename      ) { { T{filename}                   };                                                          }
 && requires(T&& t, typename T::resource_t&& resource) { { T{std::move(resource)}        };                                                          }
 && requires(T&& t, const std::string& filename      ) { { t.read(filename)              }          -> std::convertible_to<typename T::resource_t&>; }
 && requires(T&& t, const std::string& filename      ) { { t.write(filename)             }          -> std::convertible_to<bool>;                    }
 && requires(T&& t, typename T::resource_t& resource ) { { t.get()                       } noexcept -> std::convertible_to<typename T::resource_t&>; }
 && requires(T&& t, typename T::resource_t&& resource) { { t.assign(std::move(resource)) }          -> std::convertible_to<bool>;                    }
 ;

template <typename T, typename ResourceT>
struct IResource
{
    using resource_t = ResourceT;

    IResource()                            { static_assert(Resource<T>, "IResource not implemented properly"); }
    IResource(IResource&&) = default;
    IResource(const std::string& filename) { static_assert(Resource<T>, "IResource not implemented properly"); }
    IResource(resource_t&&)                { static_assert(Resource<T>, "IResource not implemented properly"); }
};

#endif
