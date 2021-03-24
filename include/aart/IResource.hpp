#ifndef AART_I_RESOURCE_H
#define AART_I_RESOURCE_H

#include <concepts>
#include <string>

#include <aart/utility.hpp>

template <typename T>
concept Resource = 
    requires(T t, Filename const& filename               ) {{ T{filename}                   };                                                       }
 && requires(T t, typename T::resource_t&& resource      ) {{ T{std::move(resource)}        };                                                       }
 && requires(T t, Filename const& filename               ) {{ t.read(filename)              } -> std::convertible_to<typename T::resource_t&>;       }
 && requires(const T t, Filename const& filename         ) {{ t.write(filename)             } -> std::convertible_to<bool>;                          }
 && requires(T t, typename T::resource_t& resource       ) {{ t.get()                       } -> std::convertible_to<typename T::resource_t&>;       }
 && requires(const T t, typename T::resource_t& resource ) {{ t.get()                       } -> std::convertible_to<const typename T::resource_t&>; }
 && requires(T t, typename T::resource_t&& resource      ) {{ t.assign(std::move(resource)) } -> std::convertible_to<bool>;                          }
 ;

template <typename T, typename ResourceT>
struct IResource
{
    using resource_t = ResourceT;
    IResource() requires(Resource<T>) = default;
};

#endif
