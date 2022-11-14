#ifndef PETSC_CPP_UTILITY_HPP
#define PETSC_CPP_UTILITY_HPP

#if defined(__cplusplus)
  #include <petsc/private/cpp/macros.hpp>
  #include <petsc/private/cpp/type_traits.hpp>

  #include <utility>
  #include <cstdint> // std::uint32_t

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

// ==========================================================================================
// compressed_pair
//
// Like std::pair except that it potentially stores both first and second as base classes if
// either or both are "empty" classes. This allows empty-base-optimization to kick in. Normally
// in C++ a structure must have a minimum memory footprint of 1 byte. For example
//
// struct Foo { }; // empty!
//
// struct Bar
// {
//   Foo f; // even though Foo is empty, member 'f' will always occupy 1 byte in memory
// };
//
// This restriction does not hold for base classes however, so changing the above declarations
// to
//
// struct Foo { }; // empty!
//
// struct Bar : Foo
// {
//
// };
//
// Results in Bar now potentially occupying no space whatsoever.
// ==========================================================================================

namespace detail
{

template <bool t_empty, bool u_empty>
struct compressed_pair_selector;

template <>
struct compressed_pair_selector<false, false> : std::integral_constant<int, 0> { };

template <>
struct compressed_pair_selector<true, false> : std::integral_constant<int, 1> { };

template <>
struct compressed_pair_selector<false, true> : std::integral_constant<int, 2> { };

template <>
struct compressed_pair_selector<true, true> : std::integral_constant<int, 3> { };

template <typename T, typename U, int selector>
class compressed_pair_impl;

// selector = 0, neither are empty, derive directly from std::pair
template <typename T, typename U>
class compressed_pair_impl<T, U, 0> : std::pair<T, U> {
  using base_type = std::pair<T, U>;

public:
  using base_type::base_type;
  using typename base_type::first_type;
  using typename base_type::second_type;

  first_type       &first() noexcept { return static_cast<base_type &>(*this).first; }
  const first_type &first() const noexcept { return static_cast<const base_type &>(*this).first; }

  second_type       &second() noexcept { return static_cast<base_type &>(*this).second; }
  const second_type &second() const noexcept { return static_cast<const base_type &>(*this).second; }
};

// selector = 1, T is empty
template <typename T, typename U>
class compressed_pair_impl<T, U, 1> : T {
  using base_type = T;

public:
  using base_type::base_type;
  using first_type  = T;
  using second_type = U;

  compressed_pair_impl() = default;

  compressed_pair_impl(first_type x, second_type y) : base_type(std::move_if_noexcept(x)), second_(std::move_if_noexcept(y)) { }

  compressed_pair_impl(second_type x) : second_(std::move_if_noexcept(x)) { }

  first_type       &first() noexcept { return *this; }
  const first_type &first() const noexcept { return *this; }

  second_type       &second() noexcept { return second_; }
  const second_type &second() const noexcept { return second_; }

private:
  second_type second_;
};

// selector = 2, U is empty
template <typename T, typename U>
class compressed_pair_impl<T, U, 2> : U {
  using base_type = U;

public:
  using base_type::base_type;
  using first_type  = T;
  using second_type = U;

  compressed_pair_impl() = default;

  compressed_pair_impl(first_type x, second_type y) : base_type(std::move_if_noexcept(y)), first_(std::move_if_noexcept(x)) { }

  compressed_pair_impl(first_type x) : first_(std::move_if_noexcept(x)) { }

  first_type       &first() noexcept { return first_; }
  const first_type &first() const noexcept { return first_; }

  second_type       &second() noexcept { return *this; }
  const second_type &second() const noexcept { return *this; }

private:
  first_type first_;
};

// selector = 3, T and U are both empty
template <typename T, typename U>
class compressed_pair_impl<T, U, 3> : T, U {
  using first_base_type  = T;
  using second_base_type = U;

public:
  using first_type  = T;
  using second_type = U;

  using first_type::first_type;
  using second_type::second_type;

  compressed_pair_impl() = default;

  compressed_pair_impl(first_type x, second_type y) : first_type(std::move_if_noexcept(x)), second_type(std::move_if_noexcept(y)) { }

  // Casts are needed to disambiguate case where T or U derive from one another, for example
  //
  // struct T { };
  // struct U : T { };
  //
  // In this case both U and T are able to satisfy "conversion" to T
  first_type       &first() noexcept { return static_cast<first_type &>(*this); }
  const first_type &first() const noexcept { return static_cast<const first_type &>(*this); }

  second_type       &second() noexcept { return static_cast<second_type &>(*this); }
  const second_type &second() const noexcept { return static_cast<const second_type &>(*this); }
};

} // namespace detail

// clang-format off
template <typename T, typename U>
class compressed_pair
  : public detail::compressed_pair_impl<
      T, U,
      detail::compressed_pair_selector<std::is_empty<T>::value, std::is_empty<U>::value>::value
    >
// clang-format on
{
  using base_type = detail::compressed_pair_impl<T, U, detail::compressed_pair_selector<std::is_empty<T>::value, std::is_empty<U>::value>::value>;

public:
  using base_type::base_type;
};

  // intel compilers don't implement empty base optimization, so these tests fail
  #if !defined(__INTEL_COMPILER) && !defined(__ICL)

namespace compressed_pair_test
{

namespace
{

struct Empty { };

static_assert(std::is_empty<Empty>::value, "");
static_assert(sizeof(Empty) == 1, "");

struct Empty2 { };

static_assert(std::is_empty<Empty2>::value, "");
static_assert(sizeof(Empty2) == 1, "");

struct NotEmpty {
  std::uint32_t d{};
};

static_assert(!std::is_empty<NotEmpty>::value, "");
static_assert(sizeof(NotEmpty) > 1, "");

struct EmptyMember {
  Empty  m{};
  Empty2 m2{};
};

static_assert(!std::is_empty<EmptyMember>::value, "");
static_assert(sizeof(EmptyMember) > 1, "");

// empty-empty should only be 1 byte since both are compressed out
static_assert(std::is_empty<compressed_pair<Empty, Empty2>>::value, "");
static_assert(sizeof(compressed_pair<Empty, Empty2>) == 1, "");

// flipping template param order changes nothing
static_assert(std::is_empty<compressed_pair<Empty2, Empty>>::value, "");
static_assert(sizeof(compressed_pair<Empty2, Empty>) == 1, "");

// empty-not_empty should be less than sum of sizes, since empty is compressed out
static_assert(!std::is_empty<compressed_pair<Empty, NotEmpty>>::value, "");
static_assert(sizeof(compressed_pair<Empty, NotEmpty>) < (sizeof(Empty) + sizeof(NotEmpty)), "");

// flipping template param order changes nothing
static_assert(!std::is_empty<compressed_pair<NotEmpty, Empty>>::value, "");
static_assert(sizeof(compressed_pair<NotEmpty, Empty>) < (sizeof(NotEmpty) + sizeof(Empty)), "");

// empty_member-not_empty should also be greater than or equal to sum of sizes (g.t. because
// potential padding) because neither is compressed away
static_assert(!std::is_empty<compressed_pair<EmptyMember, NotEmpty>>::value, "");
static_assert(sizeof(compressed_pair<EmptyMember, NotEmpty>) >= (sizeof(EmptyMember) + sizeof(NotEmpty)), "");

// flipping template param order changes nothing
static_assert(!std::is_empty<compressed_pair<NotEmpty, EmptyMember>>::value, "");
static_assert(sizeof(compressed_pair<NotEmpty, EmptyMember>) >= (sizeof(NotEmpty) + sizeof(EmptyMember)), "");

} // anonymous namespace

} // namespace compressed_pair_test

  #endif

} // namespace util

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CPP_UTILITY_HPP
