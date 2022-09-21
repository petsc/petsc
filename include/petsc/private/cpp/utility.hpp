#ifndef PETSC_CPP_UTILITY_HPP
#define PETSC_CPP_UTILITY_HPP

#if defined(__cplusplus)
  #include <petsc/private/cpp/macros.hpp>
  #include <petsc/private/cpp/type_traits.hpp>

  #include <utility>

namespace Petsc
{

namespace util
{

  #if PETSC_CPP_VERSION >= 14 // C++14
using std::exchange;
using std::integer_sequence;
using std::make_integer_sequence;
  #else
template <class T, class U = T>
inline T exchange(T &orig, U &&new_value)
{
  T old_value = std::move(orig);
  orig        = std::forward<U>(new_value);
  return old_value;
}

template <class T, T... idx>
struct integer_sequence {
  static_assert(std::is_integral<T>::value, "");

  using value_type = T;

  static constexpr std::size_t size() noexcept { return sizeof...(idx); }
};

    #ifndef __has_builtin
      #define __has_builtin(x) 0
    #endif

    #if __has_builtin(__make_integer_seq)    // clang, MSVC
template <class T, T N>
using make_integer_sequence = __make_integer_seq<integer_sequence, T, N>;
    #elif defined(__GNUC__) && __GNUC__ >= 8 // gcc
template <class T, T N>
using make_integer_sequence = integer_sequence<T, __integer_pack(N)...>;
    #else                                    // __slow__ version
namespace detail
{

template <class T, int N, T... idx>
struct make_sequence : make_sequence<T, N - 1, T(N - 1), idx...> { };

template <class T, T... idx>
struct make_sequence<T, 0, idx...> {
  using type = integer_sequence<T, idx...>;
};

} // namespace detail

template <class T, T N>
using make_integer_sequence = typename detail::make_sequence<T, int(N)>::type;
    #endif                                   // __has_builtin(__make_integer_seq)
  #endif                                     // C++14

template <std::size_t... idx>
using index_sequence = integer_sequence<std::size_t, idx...>;
template <std::size_t N>
using make_index_sequence = make_integer_sequence<std::size_t, N>;
template <class... T>
using index_sequence_for = make_index_sequence<sizeof...(T)>;

} // namespace util

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CPP_UTILITY_HPP
