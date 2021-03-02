#ifndef AART_I_RESOURCE_MANAGER_H
#define AART_I_RESOURCE_MANAGER_H

#include <concepts>
#include <string>

template <typename T>
concept ResourceManager = 
    requires(T&& t, const std::string& filename      ) { { t.read(filename)              }          -> std::convertible_to<typename T::resource_t&>; }
 && requires(T&& t, const std::string& filename      ) { { t.write(filename)             }          -> std::convertible_to<bool>;                    }
 && requires(T&& t, typename T::resource_t& resource ) { { t.getResource()               } noexcept -> std::convertible_to<typename T::resource_t&>; }
 && requires(T&& t, typename T::resource_t&& resource) { { t.assign(std::move(resource)) }          -> std::convertible_to<bool>;                    }
 ;

template <typename T, typename Resource>
struct IResourceManager
{
    using resource_t = Resource;

    IResourceManager() { static_assert(ResourceManager<T>, "IResourceManager not implemented properly"); };
    IResourceManager(IResourceManager&&) = default;
};

#endif
