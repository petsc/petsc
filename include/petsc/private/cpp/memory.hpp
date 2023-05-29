#ifndef PETSC_CPP_MEMORY_HPP
#define PETSC_CPP_MEMORY_HPP

#include <petsc/private/cpp/type_traits.hpp> // remove_extent
#include <petsc/private/cpp/utility.hpp>

#if defined(__cplusplus)
  #include <memory>

namespace Petsc
{

namespace util
{

  #if PETSC_CPP_VERSION >= 14
using std::make_unique;
  #else
namespace detail
{

// helpers shamelessly stolen from libcpp
template <class T>
struct unique_if {
  using unique_single = std::unique_ptr<T>;
};

template <class T>
struct unique_if<T[]> {
  using unique_array_unknown_bound = std::unique_ptr<T[]>;
};

template <class T, std::size_t N>
struct unique_if<T[N]> {
  using unique_array_unknown_bound = void;
};

} // namespace detail

template <class T, class... Args>
inline typename detail::unique_if<T>::unique_single make_unique(Args &&...args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <class T>
inline typename detail::unique_if<T>::unique_array_unknown_bound make_unique(std::size_t n)
{
  return std::unique_ptr<T>(new util::remove_extent_t<T>[n]());
}

template <class T, class... Args>
typename detail::unique_if<T>::unique_array_known_bound make_unique(Args &&...) = delete;
  #endif // PETSC_CPP_VERSION >= 14

  #if PETSC_CPP_VERSION >= 20
// only use std::destroy_at from C++20 onwards (even though it was introduced in C++17) since
// that makes the behavior more uniform for arrays.
using std::destroy_at;
using std::construct_at;
  #else
template <class T>
inline enable_if_t<!std::is_array<T>::value> destroy_at(T *ptr) noexcept(std::is_nothrow_destructible<T>::value)
{
  ptr->~T();
}

template <class T>
inline enable_if_t<std::is_array<T>::value> destroy_at(T *ptr)
{
  for (auto &elem : *ptr) destroy_at(std::addressof(elem));
}

template <class T, class... Args, class = decltype(::new(std::declval<void *>()) T{std::declval<Args>()...})>
inline constexpr T *construct_at(T *ptr, Args &&...args) noexcept(std::is_nothrow_constructible<T, Args...>::value)
{
  return ::new ((void *)ptr) T{std::forward<Args>(args)...};
}
  #endif

} // namespace util

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CPP_MEMORY_HPP
