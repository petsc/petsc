#pragma once

#include <petsc/private/cpp/type_traits.hpp>
#include <petsc/private/cpp/utility.hpp>

#include <tuple>

namespace Petsc
{

namespace util
{

#if PETSC_CPP_VERSION >= 14
using std::tuple_element_t;
#else
template <std::size_t I, class T>
using tuple_element_t = typename std::tuple_element<I, T>::type;
#endif

// tuple_for_each
namespace detail
{

template <std::size_t... Idx, typename T, typename F>
constexpr inline F &&tuple_for_each(index_sequence<Idx...>, T &&tuple, F &&f)
{
  using expander = int[sizeof...(Idx)];
  return (void)expander{((void)f(std::get<Idx>(std::forward<T>(tuple))), 0)...}, std::forward<F>(f);
}

template <typename T, typename F>
constexpr inline F &&tuple_for_each(index_sequence<>, T &&, F &&f) noexcept
{
  return std::forward<F>(f);
}

} // namespace detail

template <typename T, typename F>
constexpr inline F &&tuple_for_each(T &&tuple, F &&f)
{
  using seq = make_index_sequence<std::tuple_size<remove_reference_t<T>>::value>;
  return detail::tuple_for_each(seq{}, std::forward<T>(tuple), std::forward<F>(f));
}

} // namespace util

} // namespace Petsc
