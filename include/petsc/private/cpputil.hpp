#ifndef PETSC_CPPUTIL_HPP
#define PETSC_CPPUTIL_HPP

#include <petsc/private/petscimpl.h>

#if defined(__cplusplus)
#include <type_traits>
#include <tuple>

namespace Petsc
{

namespace util
{

#if __cplusplus >= 201402L // C++14
using std::conditional_t;
using std::remove_const_t;
using std::add_const_t;
using std::underlying_type_t;
using std::remove_pointer_t;
using std::add_pointer_t;
using std::index_sequence;
using std::make_index_sequence;
using std::decay_t;
using std::tuple_element_t;
#if __cplusplus >= 201703L
using std::void_t;
#else // C++17
template <class... T> using void_t = void;
#endif // C++17
#else // C++14
template <bool B, class T, class F> using conditional_t = typename std::conditional<B,T,F>::type;
template <class T> using remove_const_t    = typename std::remove_const<T>::type;
template <class T> using add_const_t       = typename std::add_const<T>::type;
template <class T> using underlying_type_t = typename std::underlying_type<T>::type;
template <class T> using remove_pointer_t  = typename std::remove_pointer<T>::type;
template <class T> using add_pointer_t     = typename std::add_pointer<T>::type;
template <class T> using decay_t           = typename std::decay<T>::type;
template <class... T> using void_t = void;
template <std::size_t I, class T> using tuple_element_t = typename std::tuple_element<I,T>::type;
// index sequence only
template <std::size_t... idx> struct index_sequence
{
  using value_type = std::size_t;

  static constexpr value_type size() noexcept { return sizeof...(idx); }
};

namespace detail
{

template <std::size_t N, std::size_t... rest>
struct index_sequence_impl : index_sequence_impl<N-1U,N-1U,rest...>
{ };

template <std::size_t... rest>
struct index_sequence_impl<0U,rest...>
{
  using type = index_sequence<rest...>;
};

} // namespace detail

template <std::size_t N> using make_index_sequence = typename detail::index_sequence_impl<N>::type;
#endif // c++14

namespace detail
{

template <typename T, typename U = _p_PetscObject>
struct is_petsc_object_impl : std::false_type { };

template <typename T> struct is_petsc_object_impl<T,PetscObject> : std::true_type { };

template <typename T>
struct is_petsc_object_impl<T,decltype(T::hdr)>
  : conditional_t<
  (!std::is_pointer<T>::value) && std::is_class<T>::value && std::is_standard_layout<T>::value,
  std::true_type,
  std::false_type
  >
{ };

} // namespace detail

template <typename T> using is_petsc_object = detail::is_petsc_object_impl<remove_pointer_t<T>>;

template <typename T>
static inline constexpr underlying_type_t<T> integral_value(T value) noexcept
{
  static_assert(std::is_enum<T>::value,"");
  return static_cast<underlying_type_t<T>>(value);
}

namespace detail
{

struct can_call_test
{
  template<typename F, typename... A>
  static decltype(std::declval<F>()(std::declval<A>()...),std::true_type()) f(int);

  template<typename F, typename... A> static std::false_type f(...);
};

} // namespace detail

template <typename F, typename... A>
struct can_call : decltype(detail::can_call_test::f<F,A...>(0))
{ };

template <typename... A, typename F>
static inline constexpr can_call<F,A...> is_callable_with(F&&) noexcept
{
  return can_call<F,A...>{};
}

template <typename... T> struct always_false : std::false_type { };

} // namespace util

template <typename T>
static inline constexpr util::remove_const_t<T>& PetscRemoveConstCast(T& object) noexcept
{
  return const_cast<util::remove_const_t<T>&>(object);
}

template <typename T>
static inline constexpr T& PetscRemoveConstCast(const T& object) noexcept
{
  return const_cast<T&>(object);
}

template <typename T>
static inline constexpr T*& PetscRemoveConstCast(const T*& object) noexcept
{
  return const_cast<T*&>(object);
}

template <typename T>
static inline constexpr util::add_const_t<T>& PetscAddConstCast(T& object) noexcept
{
  return const_cast<util::add_const_t<T>&>(std::forward<T>(object));
}

template <typename T>
static inline constexpr util::add_const_t<T>*& PetscAddConstCast(T*& object) noexcept
{
  static_assert(!std::is_const<T>::value,"");
  return const_cast<util::add_const_t<T>*&>(std::forward<T>(object));
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
static inline constexpr PetscObject& PetscObjectCast(T& object) noexcept
{
  static_assert(util::is_petsc_object<T>::value,"Did you forget to include the private header?");
  return reinterpret_cast<PetscObject&>(object);
}

template <typename T>
static inline constexpr PetscObject& PetscObjectCast(const T& object) noexcept
{
  return PetscObjectCast(PetscRemoveConstCast(object));
}

#define PETSC_ALIAS_FUNCTION__(alias,original,dispatch)			\
  template <typename... Args>						\
  static inline auto dispatch(int, Args&&... args)		\
    noexcept(noexcept(original(std::forward<Args>(args)...)))		\
    -> decltype(original(std::forward<Args>(args)...))			\
  {									\
    return original(std::forward<Args>(args)...);			\
  }									\
  template <typename... Args>						\
  static inline int dispatch(char,Args&&...)			\
  {									\
    static_assert(							\
      Petsc::util::is_callable_with<Args...>(original) &&		\
      Petsc::util::always_false<Args...>::value,			\
      "function is not callable with given arguments"			\
    );									\
    return EXIT_FAILURE;						\
  }									\
  template <typename... Args>                                           \
  PETSC_NODISCARD auto alias(Args&&... args)				\
    noexcept(noexcept(dispatch(0,std::forward<Args>(args)...)))         \
    -> decltype(dispatch(0,std::forward<Args>(args)...))                \
  {                                                                     \
    return dispatch(0,std::forward<Args>(args)...);                     \
  }

// PETSC_ALIAS_FUNCTION() - Alias a function
//
// input params:
// alias    - the new name for the function
// original - the name of the function you would like to alias
//
// notes:
// Using this macro in effect creates
//
// template <typename... T>
// auto alias(T&&... args)
// {
//   return original(std::forward<T>(args)...);
// }
//
// meaning it will transparently work for any kind of alias (including overloads).
//
// example usage:
// PETSC_ALIAS_FUNCTION(bar,foo);
#define PETSC_ALIAS_FUNCTION_(alias,original,prefix)                    \
  PETSC_ALIAS_FUNCTION__(alias,original,PetscConcat(PetscConcat(prefix,_),original))
#define PETSC_ALIAS_FUNCTION(alias,original)                            \
  PETSC_ALIAS_FUNCTION_(alias,original,PetscConcat(PetscAliasFunctionDispatch_,__LINE__))

// Similar to PETSC_ALIAS_FUNCTION() this macro creates a thin wrapper which passes all
// arguments to the target function ~except~ the last N arguments. So
//
// PETSC_ALIAS_FUNCTION_GOBBLE_NTH_ARGS(bar,foo,3);
//
// creates a function with the effect of
//
// returnType bar(argType1 arg1, argType2 arg2, ..., argTypeN argN)
// {
//   IGNORE(argN);
//   IGNORE(argN-1);
//   IGNORE(argN-2);
//   return foo(arg1,arg2,...,argN-3);
// }
//
// for you.
#define PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS_(alias,original,gobblefn,N) \
  template <typename TupleT, std::size_t... idx> static inline    \
  auto gobblefn(TupleT&& tuple, Petsc::util::index_sequence<idx...>)    \
    noexcept(noexcept(original(std::get<idx>(tuple)...)))               \
    -> decltype(original(std::get<idx>(tuple)...))                      \
  {                                                                     \
    return original(std::get<idx>(tuple)...);                           \
  }                                                                     \
  template <typename... Args>                                           \
  PETSC_NODISCARD auto alias(Args&&... args)                            \
    noexcept(noexcept(gobblefn(std::forward_as_tuple(args...),Petsc::util::make_index_sequence<sizeof...(Args)-(N)>{}))) \
    -> decltype(gobblefn(std::forward_as_tuple(args...),Petsc::util::make_index_sequence<sizeof...(Args)-(N)>{})) \
  {                                                                     \
    static_assert(std::is_integral<decltype(N)>::value,"");             \
    static_assert((N) >= 0,"");                                         \
    using seq = Petsc::util::make_index_sequence<sizeof...(Args)-(N)>;  \
    return gobblefn(std::forward_as_tuple(args...),seq{});              \
  }

#define PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS(alias,original,N)     \
  PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS_(alias,original,PetscConcat(petsc_private_gobble_,original),N)

// PETSC_CXX_COMPAT_DECL() - Helper macro to declare a C++ class member function or
// free-standing function guaranteed to be compatible with C
//
// input params:
// __VA_ARGS__ - the function declaration
//
// notes:
// Normally member functions of C++ structs or classes are not callable from C as they have an
// implicit "this" parameter tacked on the front (analogous to Pythons "self"). Static
// functions on the other hand do not have this restriction. This macro applies static to the
// function declaration as well as noexcept (as C++ exceptions escaping the C++ boundary is
// undefined behavior anyways) and [[nodiscard]].
//
// Note that the user should take care that function arguments and return type are also C
// compatible.
//
// example usage:
// class myclass
// {
// public:
//   PETSC_CXX_COMPAT_DECL(PetscErrorCode foo(int,Vec,char));
// };
//
// use this to define inline as well
//
// class myclass
// {
// public:
//   PETSC_CXX_COMPAT_DECL(PetscErrorCode foo(int a, Vec b, charc))
//   {
//     ...
//   }
// };
//
// or to define a free-standing function
//
// PETSC_CXX_COMPAT_DECL(bool bar(int x, int y))
// {
//   ...
// }
#define PETSC_CXX_COMPAT_DECL(...) PETSC_NODISCARD static __VA_ARGS__ noexcept

// PETSC_CXX_COMPAT_DEFN() - Corresponding macro to define a C++ member function declared using
// PETSC_CXX_COMPAT_DECL()
//
// input params:
// __VA_ARGS__ - the function prototype (not the body!)
//
// notes:
// prepends inline and appends noexcept to the function
//
// example usage:
// PETSC_CXX_COMPAT_DEFN(PetscErrorCode myclass::foo(int a, Vec b, char c))
// {
//   ...
// }
#define PETSC_CXX_COMPAT_DEFN(...) inline __VA_ARGS__ noexcept

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CPPUTIL_HPP
