#pragma once

#include <petsc/private/petscimpl.h> // _p_PetscObject
#include <petsc/private/cpp/macros.hpp>

#include <type_traits>

namespace Petsc
{

namespace util
{

#if PETSC_CPP_VERSION >= 14
using std::add_const_t;
using std::add_pointer_t;
using std::conditional_t;
using std::decay_t;
using std::enable_if_t;
using std::remove_const_t;
using std::remove_cv_t;
using std::remove_extent_t;
using std::remove_pointer_t;
using std::remove_reference_t;
using std::underlying_type_t;
using std::common_type_t;
#else  // C++14
template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;
template <bool B, class T, class F>
using conditional_t = typename std::conditional<B, T, F>::type;
template <class T>
using remove_const_t = typename std::remove_const<T>::type;
template <class T>
using add_const_t = typename std::add_const<T>::type;
template <class T>
using remove_cv_t = typename std::remove_cv<T>::type;
template <class T>
using underlying_type_t = typename std::underlying_type<T>::type;
template <class T>
using remove_pointer_t = typename std::remove_pointer<T>::type;
template <class T>
using add_pointer_t = typename std::add_pointer<T>::type;
template <class T>
using decay_t = typename std::decay<T>::type;
template <class T>
using remove_reference_t = typename std::remove_reference<T>::type;
template <class T>
using remove_extent_t = typename std::remove_extent<T>::type;
template <class... T>
using common_type_t = typename std::common_type<T...>::type;
#endif // C++14

#if PETSC_CPP_VERSION >= 17
using std::void_t;
using std::invoke_result_t;
using std::disjunction;
using std::conjunction;
#else
template <class...>
using void_t = void;

template <class Func, class... Args>
using invoke_result_t = typename std::result_of<Func &&(Args &&...)>::type;

template <class...>
struct disjunction : std::false_type { };
template <class B1>
struct disjunction<B1> : B1 { };
template <class B1, class... Bn>
struct disjunction<B1, Bn...> : conditional_t<bool(B1::value), B1, disjunction<Bn...>> { };

template <class...>
struct conjunction : std::true_type { };
template <class B1>
struct conjunction<B1> : B1 { };
template <class B1, class... Bn>
struct conjunction<B1, Bn...> : conditional_t<bool(B1::value), conjunction<Bn...>, B1> { };
#endif

#if PETSC_CPP_VERSION >= 20
using std::remove_cvref;
using std::remove_cvref_t;
using std::type_identity;
using std::type_identity_t;
#else
template <class T>
struct remove_cvref {
  using type = remove_cv_t<remove_reference_t<T>>;
};

template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;

template <class T>
struct type_identity {
  using type = T;
};

template <class T>
using type_identity_t = typename type_identity<T>::type;
#endif // C++20

#if PETSC_CPP_VERSION >= 23
using std::to_underlying;
#else
template <typename T>
static inline constexpr underlying_type_t<T> to_underlying(T value) noexcept
{
  static_assert(std::is_enum<T>::value, "");
  return static_cast<underlying_type_t<T>>(value);
}

#endif

template <typename... T>
struct always_false : std::false_type { };

namespace detail
{

template <typename T, typename U = _p_PetscObject>
struct is_derived_petsc_object_impl : conditional_t<!std::is_same<T, U>::value && std::is_base_of<_p_PetscObject, T>::value, std::true_type, std::false_type> { };

template <typename T>
struct is_derived_petsc_object_impl<T, decltype(T::hdr)> : conditional_t<std::is_class<T>::value && std::is_standard_layout<T>::value, std::true_type, std::false_type> { };

namespace test
{

struct Empty { };

struct IntHdr {
  int hdr;
};

struct CPetscObject {
  _p_PetscObject hdr;
  int            x;
};

struct CxxPetscObject {
  void          *x;
  _p_PetscObject hdr;
};

struct CxxDerivedPetscObject : _p_PetscObject { };

// PetscObject is not derived from itself
static_assert(!::Petsc::util::detail::is_derived_petsc_object_impl<_p_PetscObject>::value, "");
// an int is not a PetscObject
static_assert(!::Petsc::util::detail::is_derived_petsc_object_impl<int>::value, "");
static_assert(!::Petsc::util::detail::is_derived_petsc_object_impl<Empty>::value, "");
static_assert(!::Petsc::util::detail::is_derived_petsc_object_impl<IntHdr>::value, "");

// each of these should be valid in PetscObjectCast()
static_assert(::Petsc::util::detail::is_derived_petsc_object_impl<CPetscObject>::value, "");
static_assert(::Petsc::util::detail::is_derived_petsc_object_impl<CxxPetscObject>::value, "");
static_assert(::Petsc::util::detail::is_derived_petsc_object_impl<CxxDerivedPetscObject>::value, "");

} // namespace test

} // namespace detail

template <typename T>
using is_derived_petsc_object = detail::is_derived_petsc_object_impl<remove_pointer_t<decay_t<T>>>;

template <class, template <class> class>
struct is_instance : public std::false_type { };

template <class T, template <class> class U>
struct is_instance<U<T>, U> : public std::true_type { };

namespace detail
{
template <template <class> class B, class E>
struct is_crtp_base_of_impl : std::is_base_of<B<E>, E> { };

template <template <class> class B, class E, template <class> class F>
struct is_crtp_base_of_impl<B, F<E>> : disjunction<std::is_base_of<B<E>, F<E>>, std::is_base_of<B<F<E>>, F<E>>> { };
} // namespace detail

template <template <class> class B, class E>
using is_crtp_base_of = detail::is_crtp_base_of_impl<B, decay_t<E>>;

} // namespace util

} // namespace Petsc

template <typename T>
PETSC_NODISCARD inline constexpr Petsc::util::remove_const_t<T> &PetscRemoveConstCast(T &object) noexcept
{
  return const_cast<Petsc::util::remove_const_t<T> &>(object);
}

template <typename T>
PETSC_NODISCARD inline constexpr T &PetscRemoveConstCast(const T &object) noexcept
{
  return const_cast<T &>(object);
}

template <typename T>
PETSC_NODISCARD inline constexpr T *&PetscRemoveConstCast(const T *&object) noexcept
{
  return const_cast<T *&>(object);
}

template <typename T>
PETSC_NODISCARD inline constexpr Petsc::util::add_const_t<T> &PetscAddConstCast(T &object) noexcept
{
  return const_cast<Petsc::util::add_const_t<T> &>(std::forward<T>(object));
}

template <typename T>
PETSC_NODISCARD inline constexpr Petsc::util::add_const_t<T> *&PetscAddConstCast(T *&object) noexcept
{
  static_assert(!std::is_const<T>::value, "");
  return const_cast<Petsc::util::add_const_t<T> *&>(std::forward<T>(object));
}

// PetscObjectCast() - Cast an object to PetscObject
//
// input param:
// object - the object to cast
//
// output param:
// [return value] - The resulting PetscObject
//
// notes:
// This function checks that the object passed in is in fact a PetscObject, and hence requires
// the full definition of the object. This means you must include the appropriate header
// containing the _p_<object> struct definition
//
//   not available from Fortran
template <typename T>
PETSC_NODISCARD inline constexpr PetscObject PetscObjectCast(T &&object) noexcept
{
  static_assert(Petsc::util::is_derived_petsc_object<Petsc::util::decay_t<T>>::value, "If this is a PetscObject then the private definition of the struct must be visible for this to work");
  return (PetscObject)object;
}

PETSC_NODISCARD inline constexpr PetscObject PetscObjectCast(PetscObject object) noexcept
{
  return object;
}

template <typename T>
PETSC_NODISCARD inline constexpr auto PetscObjectComm(T &&obj) noexcept -> Petsc::util::enable_if_t<!std::is_same<Petsc::util::decay_t<T>, PetscObject>::value, MPI_Comm>
{
  return PetscObjectComm(PetscObjectCast(std::forward<T>(obj)));
}
