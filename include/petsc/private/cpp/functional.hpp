#ifndef PETSC_CPP_FUNCTIONAL_HPP
#define PETSC_CPP_FUNCTIONAL_HPP

#if defined(__cplusplus)
  #include <petsc/private/cpp/macros.hpp>
  #include <petsc/private/cpp/utility.hpp>     // index_sequence
  #include <petsc/private/cpp/type_traits.hpp> // decay_t
  #include <petsc/private/cpp/tuple.hpp>       // tuple_element_t

  #include <functional>

namespace Petsc
{

namespace util
{

namespace detail
{

struct can_call_test {
  template <typename F, typename... A>
  static decltype(std::declval<F>()(std::declval<A>()...), std::true_type()) f(int);

  template <typename F, typename... A>
  static std::false_type f(...);
};

// generic template
template <typename T>
struct func_traits_impl : func_traits_impl<decltype(&T::operator())> { };

// function pointers
template <typename Ret, typename... Args>
struct func_traits_impl<Ret (*)(Args...)> {
  using result_type = Ret;

  template <std::size_t ix>
  struct arg {
    using type = util::tuple_element_t<ix, std::tuple<Args...>>;
  };
};

// class-like operator()
template <typename C, typename Ret, typename... Args>
struct func_traits_impl<Ret (C::*)(Args...) const> {
  using result_type = Ret;

  template <std::size_t ix>
  struct arg {
    using type = util::tuple_element_t<ix, std::tuple<Args...>>;
  };
};

template <typename C, typename Ret, typename... Args>
struct func_traits_impl<Ret (C::*)(Args...)> {
  using result_type = Ret;

  template <std::size_t ix>
  struct arg {
    using type = util::tuple_element_t<ix, std::tuple<Args...>>;
  };
};

} // namespace detail

template <typename F, typename... A>
struct can_call : decltype(detail::can_call_test::f<F, A...>(0)) { };

template <typename... A, typename F>
inline constexpr can_call<F, A...> is_callable_with(F &&) noexcept
{
  return can_call<F, A...>{};
}

template <typename T>
struct func_traits : detail::func_traits_impl<decay_t<T>> {
  template <std::size_t idx>
  using arg_t = typename detail::func_traits_impl<decay_t<T>>::template arg<idx>::type;
};

} // namespace util

} // namespace Petsc

  #define PETSC_ALIAS_FUNCTION_WITH_PROLOGUE_AND_EPILOGUE_(alias, original, prologue, epilogue) \
    template <typename... Args> \
    PETSC_NODISCARD auto alias(Args &&...args) PETSC_DECLTYPE_NOEXCEPT_AUTO(original(std::forward<Args>(args)...)) \
    { \
      prologue; \
      auto ret = original(std::forward<Args>(args)...); \
      epilogue; \
      return ret; \
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
  #define PETSC_ALIAS_FUNCTION(alias, original) PETSC_ALIAS_FUNCTION_WITH_PROLOGUE_AND_EPILOGUE_(alias, original, ((void)0), ((void)0))

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
  #define PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS_(alias, original, gobblefn, N) \
    static_assert(std::is_integral<decltype(N)>::value && ((N) >= 0), ""); \
    template <typename TupleT, std::size_t... idx> \
    static inline auto gobblefn(TupleT &&tuple, Petsc::util::index_sequence<idx...>) PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(original(std::get<idx>(tuple)...)) \
    template <typename... Args> \
    PETSC_NODISCARD auto alias(Args &&...args) PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(gobblefn(std::forward_as_tuple(args...), Petsc::util::make_index_sequence<sizeof...(Args) - (N)>{}))

  // makes prefix_lineno_name
  #define PETSC_ALIAS_UNIQUE_NAME_INTERNAL_(a, b, c, d, e) a##b##c##d##e
  #define PETSC_ALIAS_UNIQUE_NAME_INTERNAL(prefix, name)   PETSC_ALIAS_UNIQUE_NAME_INTERNAL_(prefix, _, __LINE__, _, name)

  #define PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS(alias, original, N) PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS_(alias, original, PETSC_ALIAS_UNIQUE_NAME_INTERNAL(PetscAliasFunctionGobbleDispatch, original), N)

#endif // __cplusplus

#endif // PETSC_CPP_FUNCTIONAL_HPP
