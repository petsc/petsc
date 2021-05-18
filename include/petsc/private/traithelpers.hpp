#ifndef PETSCTRAITHELPERS_HPP
#define PETSCTRAITHELPERS_HPP

#include <petsc/private/petscimpl.h>

#if defined(__cplusplus)

// A useful template to serve as a function wrapper factory. Given a function "foo" which
// you'd like to thinly wrap as "bar", simply doing:
//
// ALIAS_FUNCTION(bar,foo);
//
// essentially creates
//
// returnType bar(argType1 arg1, argType2 arg2, ..., argTypeN argn)
// { return foo(arg1,arg2,...,argn);}
//
// for you. You may then call bar exactly as you would foo.
#if PetscDefined(HAVE_CXX_DIALECT_CXX14)
// decltype(auto) is c++14
#define PETSC_ALIAS_FUNCTION(Alias_,Original_)                          \
  template <typename... Args>                                           \
  PETSC_NODISCARD decltype(auto) Alias_(Args&&... args)                 \
  { return Original_(std::forward<Args>(args)...);}
#else
#define PETSC_ALIAS_FUNCTION(Alias_,Original_)                          \
  template <typename... Args>                                           \
  PETSC_NODISCARD auto Alias_(Args&&... args)                           \
    -> decltype(Original_(std::forward<Args>(args)...))                 \
  { return Original_(std::forward<Args>(args)...);}
#endif // PetscDefined(HAVE_CXX_DIALECT_CXX14)

#endif /* __cplusplus */

#endif /* PETSCTRAITHELPERS_HPP */
