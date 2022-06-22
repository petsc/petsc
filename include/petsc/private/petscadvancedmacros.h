#ifndef PETSCADVANCEDMACROS_H
#define PETSCADVANCEDMACROS_H

#include <petscmacros.h>

/* ------------------------------ Like petscmacros.h but advanced ------------------------------ */

#define PETSC_IF_INTERNAL_0(result_if_true,...) __VA_ARGS__
#define PETSC_IF_INTERNAL_1(result_if_true,...) result_if_true

/*
  PetscIf - Conditionally expand to the second or remaining args

  Input Parameters:
+ cond           - Preprocessor conditional
. result_if_true - Result of macro expansion if cond expands to 1
- __VA_ARGS__    - Result of macro expansion if cond expands to 0

  Notes:
  Not available from Fortran, requires variadic macro support, definition is disabled by
  defining PETSC_SKIP_VARIADIC_MACROS.

  cond must be defined and expand (not evaluate!) to either integer literal 0 or 1. Must have
  at least 1 argument for __VA_ARGS__, but it may expand empty.

  Example usage:
.vb
  void myFunction(int,char*);
  #define MY_VAR 1
  PetscIf(MY_VAR,"hello","goodbye") -> "hello"
  PetscIf(MY_VAR,myFunction,PetscExpandToNothing)(1,"hello") -> myFunction(1,"hello")

  #define MY_VAR 0
  PetscIf(MY_VAR,"hello",func<type1,type2>()) -> func<type1,type2>()
  PetscIf(MY_VAR,myFunction,PetscExpandToNothing)(1,"hello") -> *nothing*
.ve

  Level: intermediate

.seealso: `PetscIfPetscDefined()`, `PetscConcat()`, `PetscExpandToNothing()`, `PetscCompl()`
*/
#define PetscIf(cond,result_if_true,...) PetscConcat_(PETSC_IF_INTERNAL_,cond)(result_if_true,__VA_ARGS__)

/*
  PetscIfPetscDefined - Like PetscIf(), but passes cond through PetscDefined() first

  Input Parameters:
+ cond           - Condition passed to PetscDefined()
. result_if_true - Result of macro expansion if PetscDefined(cond) expands to 1
- __VA_ARGS__    - Result of macro expansion if PetscDefined(cond) expands to 0

  Notes:
  Not available from Fortran, requires variadic macro support, definition is disabled by
  defining PETSC_SKIP_VARIADIC_MACROS.

  cond must satisfy all conditions for PetscDefined(). Must have at least 1 argument for
  __VA_ARGS__, but it may expand empty.

  Example usage:
.vb
  #define PETSC_HAVE_FOO 1
  PetscIfPetscDefined(HAVE_FOO,foo,bar) -> foo

  #undef PETSC_HAVE_FOO
  PetscIfPetscDefined(HAVE_FOO,foo,bar,baz,bop) -> bar,baz,bop
.ve

  Level: intermediate

.seealso: `PetscIf()`, `PetscDefined()`, `PetscConcat()`, `PetscExpand()`, `PetscCompl()`
*/
#define PetscIfPetscDefined(cond,result_if_true,...) PetscIf(PetscDefined(cond),result_if_true,__VA_ARGS__)

#endif /* PETSCADVANCEDMACROS_H */
