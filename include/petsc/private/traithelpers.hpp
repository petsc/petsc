#ifndef PETSCTRAITHELPERS_HPP
#define PETSCTRAITHELPERS_HPP

#include <petsc/private/petscimpl.h> // for PETSC_NODISCARD

#if defined(__cplusplus)

#if __cplusplus >= 201402L // decltype(auto) is c++14
#define PETSC_ALIAS_FUNCTION_(alias,original)                           \
  template <typename... Args>                                           \
  PETSC_NODISCARD decltype(auto) alias(Args&&... args)                  \
  {                                                                     \
    return original(std::forward<Args>(args)...);                       \
  }
#else
#define PETSC_ALIAS_FUNCTION_(alias,original)                           \
  template <typename... Args>                                           \
  PETSC_NODISCARD auto alias(Args&&... args)                            \
    -> decltype(original(std::forward<Args>(args)...))                  \
  {                                                                     \
    return original(std::forward<Args>(args)...);                       \
  }
#endif // c++14

// A useful template to serve as a function wrapper factory. Given a function "foo" which
// you'd like to thinly wrap as "bar", simply doing:
//
// PETSC_ALIAS_FUNCTION(bar,foo);
//
// essentially creates
//
// returnType bar(argType1 arg1, argType2 arg2, ..., argTypeN argn)
// {
//   return foo(arg1,arg2,...,argn);
// }
//
// for you. You may then call bar exactly as you would foo.
#define PETSC_ALIAS_FUNCTION(alias,original) PETSC_ALIAS_FUNCTION_(alias,original)

#endif /* __cplusplus */

#endif /* PETSCTRAITHELPERS_HPP */
