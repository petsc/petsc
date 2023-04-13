#ifndef PETSC_PREPROCESSOR_MACROS_H
#define PETSC_PREPROCESSOR_MACROS_H

#include <petscconf.h>
#include <petscconf_poison.h> /* for PetscDefined() error checking */

/* SUBMANSEC = Sys */

#if defined(__cplusplus)
  #if __cplusplus <= 201103L
    #define PETSC_CPP_VERSION 11
  #elif __cplusplus <= 201402L
    #define PETSC_CPP_VERSION 14
  #elif __cplusplus <= 201703L
    #define PETSC_CPP_VERSION 17
  #elif __cplusplus <= 202002L
    #define PETSC_CPP_VERSION 20
  #else
    #define PETSC_CPP_VERSION 22 // current year, or date of c++2b ratification
  #endif
#endif // __cplusplus

#ifndef PETSC_CPP_VERSION
  #define PETSC_CPP_VERSION 0
#endif

#if defined(__STDC_VERSION__)
  #if __STDC_VERSION__ <= 199901L
    // C99 except that 99 is >= 11 or 17 so we shorten it to 9 instead
    #define PETSC_C_VERSION 9
  #elif __STDC_VERSION__ <= 201112L
    #define PETSC_C_VERSION 11
  #elif __STDC_VERSION__ <= 201710L
    #define PETSC_C_VERSION 17
  #else
    #define PETSC_C_VERSION 22 // current year, or date of c2b ratification
  #endif
#endif // __STDC_VERSION__

#ifndef PETSC_C_VERSION
  #define PETSC_C_VERSION 0
#endif

/* ========================================================================== */
/* This facilitates using the C version of PETSc from C++ and the C++ version from C. */
#if defined(__cplusplus)
  #define PETSC_FUNCTION_NAME PETSC_FUNCTION_NAME_CXX
#else
  #define PETSC_FUNCTION_NAME PETSC_FUNCTION_NAME_C
#endif

/* ========================================================================== */
/* Since PETSc manages its own extern "C" handling users should never include PETSc include
 * files within extern "C". This will generate a compiler error if a user does put the include
 * file within an extern "C".
 */
#if defined(__cplusplus)
void assert_never_put_petsc_headers_inside_an_extern_c(int);
void assert_never_put_petsc_headers_inside_an_extern_c(double);
#endif

#if defined(__cplusplus)
  #define PETSC_RESTRICT PETSC_CXX_RESTRICT
#else
  #define PETSC_RESTRICT restrict
#endif

#define PETSC_INLINE        PETSC_DEPRECATED_MACRO("GCC warning \"PETSC_INLINE is deprecated (since version 3.17)\"") inline
#define PETSC_STATIC_INLINE PETSC_DEPRECATED_MACRO("GCC warning \"PETSC_STATIC_INLINE is deprecated (since version 3.17)\"") static inline

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES) /* For Win32 shared libraries */
  #define PETSC_DLLEXPORT __declspec(dllexport)
  #define PETSC_DLLIMPORT __declspec(dllimport)
  #define PETSC_VISIBILITY_INTERNAL
#elif defined(__cplusplus) && defined(PETSC_USE_VISIBILITY_CXX)
  #define PETSC_DLLEXPORT           __attribute__((visibility("default")))
  #define PETSC_DLLIMPORT           __attribute__((visibility("default")))
  #define PETSC_VISIBILITY_INTERNAL __attribute__((visibility("hidden")))
#elif !defined(__cplusplus) && defined(PETSC_USE_VISIBILITY_C)
  #define PETSC_DLLEXPORT           __attribute__((visibility("default")))
  #define PETSC_DLLIMPORT           __attribute__((visibility("default")))
  #define PETSC_VISIBILITY_INTERNAL __attribute__((visibility("hidden")))
#else
  #define PETSC_DLLEXPORT
  #define PETSC_DLLIMPORT
  #define PETSC_VISIBILITY_INTERNAL
#endif

#if defined(petsc_EXPORTS) /* CMake defines this when building the shared library */
  #define PETSC_VISIBILITY_PUBLIC PETSC_DLLEXPORT
#else /* Win32 users need this to import symbols from petsc.dll */
  #define PETSC_VISIBILITY_PUBLIC PETSC_DLLIMPORT
#endif

/* Functions tagged with PETSC_EXTERN in the header files are always defined as extern "C" when
 * compiled with C++ so they may be used from C and are always visible in the shared libraries
 */
#if defined(__cplusplus)
  #define PETSC_EXTERN         extern "C" PETSC_VISIBILITY_PUBLIC
  #define PETSC_EXTERN_TYPEDEF extern "C"
  #define PETSC_INTERN         extern "C" PETSC_VISIBILITY_INTERNAL
#else
  #define PETSC_EXTERN extern PETSC_VISIBILITY_PUBLIC
  #define PETSC_EXTERN_TYPEDEF
  #define PETSC_INTERN extern PETSC_VISIBILITY_INTERNAL
#endif

#if defined(PETSC_USE_SINGLE_LIBRARY)
  #define PETSC_SINGLE_LIBRARY_INTERN PETSC_INTERN
#else
  #define PETSC_SINGLE_LIBRARY_INTERN PETSC_EXTERN
#endif

#if !defined(__has_feature)
  #define __has_feature(x) 0
#endif

/*MC
  PetscHasAttribute - Determine whether a particular __attribute__ is supported by the compiler

  Synopsis:
  #include <petscmacros.h>
  int PetscHasAttribute(name)

  Input Parameter:
. name - The name of the attribute to test

  Notes:
  name should be identical to what you might pass to the __attribute__ declaration itself --
  plain, unbroken text.

  As `PetscHasAttribute()` is wrapper over the function-like macro `__has_attribute()`, the
  exact type and value returned is implementation defined. In practice however, it usually
  returns `1` if the attribute is supported and `0` if the attribute is not supported.

  Example Usage:
  Typical usage is using the preprocessor

.vb
  #if PetscHasAttribute(always_inline)
  #  define MY_ALWAYS_INLINE __attribute__((always_inline))
  #else
  #  define MY_ALWAYS_INLINE
  #endif

  void foo(void) MY_ALWAYS_INLINE;
.ve

  but it can also be used in regular code

.vb
  if (PetscHasAttribute(some_attribute)) {
    foo();
  } else {
    bar();
  }
.ve

  Level: intermediate

.seealso: `PetscHasBuiltin()`, `PetscDefined()`, `PetscLikely()`, `PetscUnlikely()`,
`PETSC_ATTRIBUTE_FORMAT`, `PETSC_ATTRIBUTE_MAY_ALIAS`
M*/
#if !defined(__has_attribute)
  #define __has_attribute(x) 0
#endif
#define PetscHasAttribute(name) __has_attribute(name)

/*MC
  PetscHasBuiltin - Determine whether a particular builtin method is supported by the compiler

  Synopsis:
  #include <petscmacros.h>
  int PetscHasBuiltin(name)

  Input Parameter:
. name - the name of the builtin routine

  Notes:
  Evaluates to `1` if the builtin is supported and `0` otherwise. Note the term "evaluates"
  (vs "expands") is deliberate; even though `PetscHasBuiltin()` is a macro the underlying
  detector is itself is a compiler extension with implementation-defined return type and
  semantics. Some compilers implement it as a macro, others as a compiler function. In practice
  however, all supporting compilers return an integer boolean as described.

  Example Usage:
  Typical usage is in preprocessor directives

.vb
  #if PetscHasBuiltin(__builtin_trap)
  __builtin_trap();
  #else
  abort();
  #endif
.ve

  But it may also be used in regular code

.vb
  if (PetscHasBuiltin(__builtin_alloca)) {
    foo();
  } else {
    bar();
  }
.ve

  Level: intermediate

.seealso: `PetscHasAttribute()`, `PetscAssume()`
M*/
#if !defined(__has_builtin)
  #define __has_builtin(x) 0
#endif
// clangs __has_builtin prior to clang 10 did not properly handle non-function builtins such as
// __builtin_types_compatible_p which take types or other non-functiony things as
// arguments. The correct way to detect these then is to use __is_identifier (also a clang
// extension). GCC has always worked as expected. see https://stackoverflow.com/a/45043153
#if defined(__clang__) && defined(__clang_major__) && (__clang_major__ < 10) && defined(__is_identifier)
  #define PetscHasBuiltin(name) __is_identifier(name)
#else
  #define PetscHasBuiltin(name) __has_builtin(name)
#endif

#if !defined(PETSC_SKIP_ATTRIBUTE_MPI_TYPE_TAG)
  /*
   Support for Clang (>=3.2) matching type tag arguments with void* buffer types.
   This allows the compiler to detect cases where the MPI datatype argument passed to a MPI routine
   does not match the actual type of the argument being passed in
*/
  #if PetscHasAttribute(pointer_with_type_tag)
    #define PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(bufno, typeno) __attribute__((pointer_with_type_tag(MPI, bufno, typeno)))
  #endif

  #if PetscHasAttribute(type_tag_for_datatype)
    #define PETSC_ATTRIBUTE_MPI_TYPE_TAG(type)                   __attribute__((type_tag_for_datatype(MPI, type)))
    #define PETSC_ATTRIBUTE_MPI_TYPE_TAG_LAYOUT_COMPATIBLE(type) __attribute__((type_tag_for_datatype(MPI, type, layout_compatible)))
  #endif
#endif // PETSC_SKIP_ATTRIBUTE_MPI_TYPE_TAG

#ifndef PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE
  #define PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(bufno, typeno)
#endif

#ifndef PETSC_ATTRIBUTE_MPI_TYPE_TAG
  #define PETSC_ATTRIBUTE_MPI_TYPE_TAG(type)
#endif

#ifndef PETSC_ATTRIBUTE_MPI_TYPE_TAG_LAYOUT_COMPATIBLE
  #define PETSC_ATTRIBUTE_MPI_TYPE_TAG_LAYOUT_COMPATIBLE(type)
#endif

/*MC
  PETSC_ATTRIBUTE_FORMAT - Indicate to the compiler that specified arguments should be treated
  as format specifiers and checked for validity

  Synopsis:
  #include <petscmacros.h>
  <attribute declaration> PETSC_ATTRIBUTE_FORMAT(int strIdx, int vaArgIdx)

  Input Parameters:
+ strIdx   - The (1-indexed) location of the format string in the argument list
- vaArgIdx - The (1-indexed) location of the first formattable argument in the argument list

  Level: developer

  Notes:
  This function attribute causes the compiler to issue warnings when the format specifier does
  not match the type of the variable that will be formatted, or when there exists a mismatch
  between the number of format specifiers and variables to be formatted. It is safe to use this
  macro if your compiler does not support format specifier checking (though this is
  exceeedingly rare).

  Both `strIdx` and `vaArgIdx` must be compile-time constant integer literals and cannot have the
  same value.

  The arguments to be formatted (and therefore checked by the compiler) must be "contiguous" in
  the argument list, that is, there is no way to indicate gaps which should not be checked.

  Definition is suppressed by defining `PETSC_SKIP_ATTRIBUTE_FORMAT` prior to including PETSc
  header files. In this case the macro will expand empty.

  Example Usage:
.vb
  // format string is 2nd argument, variable argument list containing args is 3rd argument
  void my_printf(void *obj, const char *fmt_string, ...) PETSC_ATTRIBUTE_FORMAT(2,3)

  int    x = 1;
  double y = 50.0;

  my_printf(NULL,"%g",x);      // WARNING, format specifier does not match for 'int'!
  my_printf(NULL,"%d",x,y);    // WARNING, more arguments than format specifiers!
  my_printf(NULL,"%d %g",x,y); // OK
.ve

.seealso: `PETSC_ATTRIBUTE_COLD`, `PetscHasAttribute()`
M*/
#if PetscHasAttribute(format) && !defined(PETSC_SKIP_ATTRIBUTE_FORMAT)
  #define PETSC_ATTRIBUTE_FORMAT(strIdx, vaArgIdx) __attribute__((format(printf, strIdx, vaArgIdx)))
#else
  #define PETSC_ATTRIBUTE_FORMAT(strIdx, vaArgIdx)
#endif

/*MC
  PETSC_ATTRIBUTE_COLD - Indicate to the compiler that a function is very unlikely to be
  executed

  Level: intermediate

  Notes:
  The marked function is often optimized for size rather than speed and may be grouped alongside
  other equally frigid routines improving code locality of lukewarm or hotter parts of program.

  The paths leading to cold functions are usually automatically marked as unlikely by the
  compiler. It may thus be useful to mark functions used to handle unlikely conditions -- such
  as error handlers -- as cold to improve optimization of the surrounding temperate functions.

  Example Usage:
.vb
  void my_error_handler(...) PETSC_ATTRIBUTE_COLD;

  if (temperature < 0) {
    return my_error_handler(...); // chilly!
  }
.ve

.seealso: `PetscUnlikely()`, `PetscUnlikelyDebug()`, `PetscLikely()`, `PetscLikelyDebug()`,
          `PetscUnreachable()`, `PETSC_ATTRIBUTE_FORMAT`
M*/
#if PetscHasAttribute(__cold__)
  #define PETSC_ATTRIBUTE_COLD __attribute__((__cold__))
#elif PetscHasAttribute(cold) /* some implementations (old gcc) use no underscores */
  #define PETSC_ATTRIBUTE_COLD __attribute__((cold))
#else
  #define PETSC_ATTRIBUTE_COLD
#endif

/*MC
  PETSC_ATTRIBUTE_MAY_ALIAS - Indicate to the compiler that a type is not
  subjected to type-based alias analysis, but is instead assumed to be able to
  alias any other type of objects

  Example Usage:
.vb
  typedef PetscScalar PetscScalarAlias PETSC_ATTRIBUTE_MAY_ALIAS;

  PetscReal        *pointer;
  PetscScalarAlias *other_pointer = reinterpret_cast<PetscScalarAlias *>(pointer);
.ve

  Level: advanced

.seealso: `PetscHasAttribute()`
M*/
#if PetscHasAttribute(may_alias) && !defined(PETSC_SKIP_ATTRIBUTE_MAY_ALIAS)
  #define PETSC_ATTRIBUTE_MAY_ALIAS __attribute__((may_alias))
#else
  #define PETSC_ATTRIBUTE_MAY_ALIAS
#endif

/*MC
  PETSC_NULLPTR - Standard way of indicating a null value or pointer

  No Fortran Support

  Level: beginner

  Notes:
  Equivalent to `NULL` in C source, and `nullptr` in C++ source. Note that for the purposes of
  interoperability between C and C++, setting a pointer to `PETSC_NULLPTR` in C++ is functonially
  equivalent to setting the same pointer to `NULL` in C. That is to say that the following
  expressions are equivalent\:

.vb
  ptr == PETSC_NULLPTR
  ptr == NULL
  ptr == 0
  !ptr

  ptr = PETSC_NULLPTR
  ptr = NULL
  ptr = 0
.ve

  and for completeness' sake\:

.vb
  PETSC_NULLPTR == NULL
.ve

  Example Usage:
.vb
  // may be used in place of '\0' or other such teminators in the definition of char arrays
  const char *const MyEnumTypes[] = {
    "foo",
    "bar",
    PETSC_NULLPTR
  };

  // may be used to nullify objects
  PetscObject obj = PETSC_NULLPTR;

  // may be used in any function expecting NULL
  PetscInfo(PETSC_NULLPTR,"Lorem Ipsum Dolor");
.ve

  Developer Notes:
  `PETSC_NULLPTR` must be used in place of `NULL` in all C++ source files. Using `NULL` in source
  files compiled with a C++ compiler may lead to unexpected side-effects in function overload
  resolution and/or compiler warnings.

.seealso: `PETSC_CONSTEXPR_14`, `PETSC_NODISCARD`
M*/

/*MC
  PETSC_CONSTEXPR_14 - C++14 constexpr

  No Fortran Support

  Level: beginner

  Notes:
  Equivalent to `constexpr` when using a C++ compiler that supports C++14. Expands to nothing
  if the C++ compiler does not support C++14 or when not compiling with a C++ compiler. Note
  that this cannot be used in cases where an empty expansion would result in invalid code. It
  is safe to use this in C source files.

  Example Usage:
.vb
  PETSC_CONSTEXPR_14 int factorial(int n)
  {
    int r = 1;

    do {
      r *= n;
    } while (--n);
    return r;
  }
.ve

.seealso: `PETSC_NULLPTR`, `PETSC_NODISCARD`
M*/

/*MC
  PETSC_NODISCARD - Mark the return value of a function as non-discardable

  Not available in Fortran

  Level: beginner

  Notes:
  Hints to the compiler that the return value of a function must be captured. A diagnostic may
  (but is not required to) be emitted if the value is discarded. It is safe to use this in both
  C and C++ source files.

  Example Usage:
.vb
  class Foo
  {
    int x;

  public:
    PETSC_NODISCARD Foo(int y) : x(y) { }
  };

  PETSC_NODISCARD int factorial(int n)
  {
    return n <= 1 ? 1 : (n * factorial(n - 1));
  }

  auto x = factorial(10); // OK, capturing return value
  factorial(10);          // Warning: ignoring return value of function declared 'nodiscard'

  auto f = Foo(x); // OK, capturing constructed object
  Foo(x);          // Warning: Ignoring temporary created by a constructor declared 'nodiscard'
.ve

.seealso: `PETSC_NULLPTR`, `PETSC_CONSTEXPR_14`
M*/

/* C++11 features */
#if defined(__cplusplus) || (PETSC_C_VERSION >= 23)
  #define PETSC_NULLPTR nullptr
#else
  #define PETSC_NULLPTR NULL
#endif

/* C++14 features */
#if PETSC_CPP_VERSION >= 14
  #define PETSC_CONSTEXPR_14 constexpr
#else
  #define PETSC_CONSTEXPR_14
#endif

/* C++17 features */
#if PETSC_CPP_VERSION >= 17
  #define PETSC_CONSTEXPR_17 constexpr
#else
  #define PETSC_CONSTEXPR_17
#endif

#if (PETSC_CPP_VERSION >= 17) || (PETSC_C_VERSION >= 23)
  #define PETSC_NODISCARD [[nodiscard]]
#elif PetscHasAttribute(warn_unused_result)
  #define PETSC_NODISCARD __attribute__((warn_unused_result))
#else
  #define PETSC_NODISCARD
#endif

#include <petscversion.h>
#define PETSC_AUTHOR_INFO "       The PETSc Team\n    petsc-maint@mcs.anl.gov\n https://petsc.org/\n"

/* designated initializers since C99 and C++20, MSVC never supports them though */
#if defined(_MSC_VER) || (defined(__cplusplus) && (PETSC_CPP_VERSION < 20))
  #define PetscDesignatedInitializer(name, ...) __VA_ARGS__
#else
  #define PetscDesignatedInitializer(name, ...) .name = __VA_ARGS__
#endif

/*MC
  PetscUnlikely - Hints the compiler that the given condition is usually false

  Synopsis:
  #include <petscmacros.h>
  bool PetscUnlikely(bool cond)

  Not Collective; No Fortran Support

  Input Parameter:
. cond - Boolean expression

  Level: advanced

  Notes:
  This returns the same truth value, it is only a hint to compilers that the result of cond is
  unlikely to be true.

  Example usage:
.vb
  if (PetscUnlikely(cond)) {
    foo(); // cold path
  } else {
    bar(); // hot path
  }
.ve

.seealso: `PetscLikely()`, `PetscUnlikelyDebug()`, `PetscCall()`, `PetscDefined()`, `PetscHasAttribute()`,
          `PETSC_ATTRIBUTE_COLD`
M*/

/*MC
  PetscLikely - Hints the compiler that the given condition is usually true

  Synopsis:
  #include <petscmacros.h>
  bool PetscLikely(bool cond)

  Not Collective; No Fortran Support

  Input Parameter:
. cond - Boolean expression

  Level: advanced

  Notes:
  This returns the same truth value, it is only a hint to compilers that the result of cond is
  likely to be true.

  Example usage:
.vb
  if (PetscLikely(cond)) {
    foo(); // hot path
  } else {
    bar(); // cold path
  }
.ve

.seealso: `PetscUnlikely()`, `PetscDefined()`, `PetscHasAttribute()`
          `PETSC_ATTRIBUTE_COLD`
M*/
#if defined(PETSC_HAVE_BUILTIN_EXPECT)
  #define PetscUnlikely(cond) __builtin_expect(!!(cond), 0)
  #define PetscLikely(cond)   __builtin_expect(!!(cond), 1)
#else
  #define PetscUnlikely(cond) (cond)
  #define PetscLikely(cond)   (cond)
#endif

/*MC
  PetscUnreachable - Indicate to the compiler that a code-path is logically unreachable

  Synopsis:
  #include <petscmacros.h>
  void PetscUnreachable(void)

  Level: advanced

  Notes:
  Indicates to the compiler (usually via some built-in) that a particular code path is always
  unreachable. Behavior is undefined if this function is ever executed, the user can expect an
  unceremonious crash.

  Example usage:
  Useful in situations such as switches over enums where not all enumeration values are
  explicitly covered by the switch

.vb
  typedef enum {RED, GREEN, BLUE} Color;

  int foo(Color c)
  {
    // it is known to programmer (or checked previously) that c is either RED or GREEN
    // but compiler may not be able to deduce this and/or emit spurious warnings
    switch (c) {
      case RED:
        return bar();
      case GREEN:
        return baz();
      default:
        PetscUnreachable(); // program is ill-formed if executed
    }
  }
.ve

.seealso: `SETERRABORT()`, `PETSCABORT()`, `PETSC_ATTRIBUTE_COLD`, `PetscAssume()`
M*/
#if PETSC_CPP_VERSION >= 23
  #include <utility>
  #define PetscUnreachable() std::unreachable()
#elif defined(__GNUC__)
  /* GCC 4.8+, Clang, Intel and other compilers compatible with GCC (-std=c++0x or above) */
  #define PetscUnreachable() __builtin_unreachable()
#elif defined(_MSC_VER) /* MSVC */
  #define PetscUnreachable() __assume(0)
#else /* ??? */
  #define PetscUnreachable() SETERRABORT(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Code path explicitly marked as unreachable executed")
#endif

/*MC
  PetscAssume - Indicate to the compiler a condition that is defined to be true

  Synopsis:
  #include <petscmacros.h>
  void PetscAssume(bool cond)

  Input Parameter:
. cond - Boolean expression

  Level: advanced

  Notes:
  If supported by the compiler, `cond` is used to inform the optimizer of an invariant
  truth. The argument itself is never evaluated, so any side effects of the expression will be
  discarded. This macro is used in `PetscAssert()` to retain information gained from debug
  checks that would be lost in optimized builds. For example\:

.vb
  PetscErrorCode foo(PetscInt x) {

    PetscAssert(x >= 0, ...);
  }
.ve

  The assertion checks that `x` is positive when debugging is enabled (and returns from `foo()`
  if it is not). This implicitly informs the optimizer that `x` cannot be negative. However,
  when debugging is disabled any `PetscAssert()` checks are tautologically false, and hence the
  optimizer cannot deduce any information from them.

  Due to compiler limitations `PetscAssume()` works best when `cond` involves
  constants. Certain compilers do not yet propagate symbolic inequalities i.e.\:

.vb
  int a, b, var_five;

  // BEST, all supporting compilers will understand a cannot be >= 5
  PetscAssume(a < 5);

   // OK, some compilers may understand that a cannot be >= 5
  PetscAssume(a <= b && b < 5);

   // WORST, most compilers will not get the memo
  PetscAssume(a <= b && b < var_five);
.ve

  If the condition is violated at runtime then behavior is wholly undefined. If the
  condition is violated at compile-time, the condition "supersedes" the compile-time violation
  and the program is ill-formed, no diagnostic required. For example consider the following\:

.vb
  PetscInt x = 0;

  PetscAssume(x != 0);
  if (x == 0) {
    x += 10;
  } else {
    popen("rm -rf /", "w");
  }
.ve

  Even though `x` is demonstrably `0` the compiler may opt to\:

  - emit an unconditional `popen("rm -rf /", "w")`
  - ignore `PetscAssume()` altogether and emit the correct path of `x += 10`
  - reformat the primary disk partition

.seealso: `PetscAssert()`
M*/
#if PETSC_CPP_VERSION >= 23
  #define PetscAssume(...) [[assume(__VA_ARGS__)]]
#elif defined(_MSC_VER) // msvc
  #define PetscAssume(...) __assume(__VA_ARGS__)
#elif defined(__clang__) && PetscHasBuiltin(__builtin_assume) // clang
  #define PetscAssume(...) \
    do { \
      _Pragma("clang diagnostic push"); \
      _Pragma("clang diagnostic ignored \"-Wassume\""); \
      __builtin_assume(__VA_ARGS__); \
      _Pragma("clang diagnostic pop"); \
    } while (0)
#else // gcc (and really old clang)
  // gcc does not have its own __builtin_assume() intrinsic. One could fake it via
  //
  // if (PetscUnlikely(!cond)) PetscUnreachable();
  //
  // but this it unsavory because the side effects of cond are not guaranteed to be
  // discarded. Though in most circumstances gcc will optimize out the if (because any evaluation
  // for which cond is false would be undefined results in undefined behavior anyway) it cannot
  // always do so. This is especially the case for opaque or non-inline function calls:
  //
  // extern int bar(int);
  //
  // int foo(int x) {
  //   PetscAssume(bar(x) == 2);
  //   if (bar(x) == 2) {
  //     return 1;
  //   } else {
  //     return 0;
  //   }
  // }
  //
  // Here gcc would (if just using builtin_expect()) emit 2 calls to bar(). Note we still have
  // cond "tested" in the condition, but this is done to silence unused-but-set variable warnings
  #define PetscAssume(...) \
    do { \
      if (0 && (__VA_ARGS__)) PetscUnreachable(); \
    } while (0)
#endif

/*MC
  PetscExpand - Expand macro argument

  Synopsis:
  #include <petscmacros.h>
  <macro-expansion> PetscExpand(x)

  Input Parameter:
. x - The preprocessor token to expand

  Level: beginner

.seealso: `PetscStringize()`, `PetscConcat()`
M*/
#define PetscExpand_(...) __VA_ARGS__
#define PetscExpand(...)  PetscExpand_(__VA_ARGS__)

/*MC
  PetscStringize - Stringize a token

  Synopsis:
  #include <petscmacros.h>
  const char* PetscStringize(x)

  No Fortran Support

  Input Parameter:
. x - The token you would like to stringize

  Output Parameter:
. <return-value> - The string representation of `x`

  Level: beginner

  Note:
  `PetscStringize()` expands `x` before stringizing it, if you do not wish to do so, use
  `PetscStringize_()` instead.

  Example Usage:
.vb
  #define MY_OTHER_VAR hello there
  #define MY_VAR       MY_OTHER_VAR

  PetscStringize(MY_VAR)  -> "hello there"
  PetscStringize_(MY_VAR) -> "MY_VAR"

  int foo;
  PetscStringize(foo)  -> "foo"
  PetscStringize_(foo) -> "foo"
.ve

.seealso: `PetscConcat()`, `PetscExpandToNothing()`, `PetscExpand()`
M*/
#define PetscStringize_(...) #__VA_ARGS__
#define PetscStringize(...)  PetscStringize_(__VA_ARGS__)

/*MC
  PetscConcat - Concatenate two tokens

  Synopsis:
  #include <petscmacros.h>
  <macro-expansion> PetscConcat(x, y)

  No Fortran Support

  Input Parameters:
+ x - First token
- y - Second token

  Level: beginner

  Note:
  `PetscConcat()` will expand both arguments before pasting them together, use `PetscConcat_()`
  if you don't want to expand them.

  Example usage:
.vb
  PetscConcat(hello,there) -> hellothere

  #define HELLO hello
  PetscConcat(HELLO,there)  -> hellothere
  PetscConcat_(HELLO,there) -> HELLOthere
.ve

.seealso: `PetscStringize()`, `PetscExpand()`
M*/
#define PetscConcat_(x, y) x##y
#define PetscConcat(x, y)  PetscConcat_(x, y)

#define PETSC_INTERNAL_COMPL_0 1
#define PETSC_INTERNAL_COMPL_1 0

/*MC
  PetscCompl - Expands to the integer complement of its argument

  Synopsis:
  #include <petscmacros.h>
  int PetscCompl(b)

  No Fortran Support

  Input Parameter:
. b - Preprocessor variable, must expand to either integer literal 0 or 1

  Output Parameter:
. <return-value> - Either integer literal 0 or 1

  Level: beginner

  Notes:
  Expands to integer literal 0 if b expands to 1, or integer literal 1 if b expands to
  0. Behaviour is undefined if b expands to anything else. PetscCompl() will expand its
  argument before returning the complement.

  This macro can be useful for negating `PetscDefined()` inside macros e.g.

$ #define PETSC_DONT_HAVE_FOO PetscCompl(PetscDefined(HAVE_FOO))

  Example usage:
.vb
  #define MY_VAR 1
  PetscCompl(MY_VAR) -> 0

  #undef  MY_VAR
  #define MY_VAR 0
  PetscCompl(MY_VAR) -> 1
.ve

.seealso: `PetscConcat()`, `PetscDefined()`
M*/
#define PetscCompl(b) PetscConcat_(PETSC_INTERNAL_COMPL_, PetscExpand(b))

/*MC
  PetscDefined - Determine whether a boolean macro is defined

  No Fortran Support

  Synopsis:
  #include <petscmacros.h>
  int PetscDefined(def)

  Input Parameter:
. def - PETSc-style preprocessor variable (without PETSC_ prepended!)

  Output Parameter:
. <return-value> - Either integer literal 0 or 1

  Level: intermediate

  Notes:
  `PetscDefined()` returns 1 if and only if "PETSC_ ## def" is defined (but empty) or defined to
  integer literal 1. In all other cases, `PetscDefined()` returns integer literal 0. Therefore
  this macro should not be used if its argument may be defined to a non-empty value other than
  1.

  The prefix "PETSC_" is automatically prepended to def. To avoid prepending "PETSC_", say to
  add custom checks in user code, one should use `PetscDefined_()`.

$ #define FooDefined(d) PetscDefined_(PetscConcat(FOO_,d))

  Developer Notes:
  Getting something that works in C and CPP for an arg that may or may not be defined is
  tricky. Here, if we have "#define PETSC_HAVE_BOOGER 1" we match on the placeholder define,
  insert the "0," for arg1 and generate the triplet (0, 1, 0). Then the last step cherry picks
  the 2nd arg (a one). When PETSC_HAVE_BOOGER is not defined, we generate a (... 1, 0) pair,
  and when the last step cherry picks the 2nd arg, we get a zero.

  Our extra expansion via PetscDefined__take_second_expand() is needed with MSVC, which has a
  nonconforming implementation of variadic macros.

  Example Usage:
  Suppose you would like to call either "foo()" or "bar()" depending on whether PETSC_USE_DEBUG
  is defined then

.vb
  #if PetscDefined(USE_DEBUG)
    foo();
  #else
    bar();
  #endif

  // or alternatively within normal code
  if (PetscDefined(USE_DEBUG)) {
    foo();
  } else {
    bar();
  }
.ve

  is equivalent to

.vb
  #if defined(PETSC_USE_DEBUG)
  #  if MY_DETECT_EMPTY_MACRO(PETSC_USE_DEBUG) // assuming you have such a macro
       foo();
  #   elif PETSC_USE_DEBUG == 1
       foo();
  #   else
       bar();
  #  endif
  #else
  bar();
  #endif
.ve

.seealso: `PetscHasAttribute()`, `PetscUnlikely()`, `PetscLikely()`, `PetscConcat()`,
          `PetscExpandToNothing()`, `PetscCompl()`
M*/
#define PetscDefined_arg_1                                    shift,
#define PetscDefined_arg_                                     shift,
#define PetscDefined__take_second_expanded(ignored, val, ...) val
#define PetscDefined__take_second_expand(args)                PetscDefined__take_second_expanded args
#define PetscDefined__take_second(...)                        PetscDefined__take_second_expand((__VA_ARGS__))
#define PetscDefined__(arg1_or_junk)                          PetscDefined__take_second(arg1_or_junk 1, 0, at_)
#define PetscDefined_(value)                                  PetscDefined__(PetscConcat_(PetscDefined_arg_, value))
#define PetscDefined(def)                                     PetscDefined_(PetscConcat(PETSC_, def))

/*MC
  PetscUnlikelyDebug - Hints the compiler that the given condition is usually false, eliding
  the check in optimized mode

  No Fortran Support

  Synopsis:
  #include <petscmacros.h>
  bool PetscUnlikelyDebug(bool cond)

  Not Collective

  Input Parameter:
. cond - Boolean expression

  Level: advanced

  Note:
  This returns the same truth value, it is only a hint to compilers that the result of `cond` is
  likely to be false. When PETSc is compiled in optimized mode this will always return
  false. Additionally, `cond` is guaranteed to not be evaluated when PETSc is compiled in
  optimized mode.

  Example usage:
  This routine is shorthand for checking both the condition and whether PetscDefined(USE_DEBUG)
  is true. So

.vb
  if (PetscUnlikelyDebug(cond)) {
    foo();
  } else {
    bar();
  }
.ve

  is equivalent to

.vb
  if (PetscDefined(USE_DEBUG)) {
    if (PetscUnlikely(cond)) {
      foo();
    } else {
      bar();
    }
  } else {
    bar();
  }
.ve

.seealso: `PetscUnlikely()`, `PetscLikely()`, `PetscCall()`, `SETERRQ`
M*/
#define PetscUnlikelyDebug(cond) (PetscDefined(USE_DEBUG) && PetscUnlikely(cond))

#if defined(PETSC_CLANG_STATIC_ANALYZER)
  // silence compiler warnings when using -pedantic, this is only used by the linter and it cares
  // not what ISO C allows
  #define PetscMacroReturns_(retexpr, ...) \
    __extension__({ \
      __VA_ARGS__; \
      retexpr; \
    })
#else
  #define PetscMacroReturns_(retexpr, ...) \
    retexpr; \
    do { \
      __VA_ARGS__; \
    } while (0)
#endif

/*MC
  PetscExpandToNothing - Expands to absolutely nothing

  No Fortran Support

  Synopsis:
  #include <petscmacros.h>
  void PetscExpandToNothing(...)

  Input Parameter:
. __VA_ARGS__ - Anything at all

  Level: beginner

  Note:
  Must have at least 1 parameter.

  Example usage:
.vb
  PetscExpandToNothing(a,b,c) -> *nothing*
.ve

.seealso: `PetscConcat()`, `PetscDefined()`, `PetscStringize()`, `PetscExpand()`
M*/
#define PetscExpandToNothing(...)

/*MC
  PetscMacroReturns - Define a macro body that returns a value

  Synopsis:
  #include <petscmacros.h>
  return_type PetscMacroReturns(return_type retexpr, ...)

  Input Parameters:
+ retexpr     - The value or expression that the macro should return
- __VA_ARGS__ - The body of the macro

  Level: intermediate

  Notes:
  Due to limitations of the C-preprocessor retexpr cannot depend on symbols declared in the
  body of the macro and should not depend on values produced as a result of the expression. The
  user should not assume that the result of this macro is equivalent to a single logical source
  line. It is not portable to use macros defined using this one in conditional or loop bodies
  without enclosing them in curly braces\:

.vb
  #define FOO(arg1) PetscMacroReturns(0,arg1+=10) // returns 0

  int err,x = 10;

  if (...) err = FOO(x);      // ERROR, body of FOO() executed outside the if statement
  if (...) { err = FOO(x); }  // OK

  for (...) err = FOO(x);     // ERROR, body of FOO() executed outside the loop
  for (...) { err = FOO(x); } // OK
.ve

  It is also not portable to use this macro directly inside function call, conditional, loop,
  or switch statements\:

.vb
  extern void bar(int);

  int ret = FOO(x);

  bar(FOO(x)); // ERROR, may not compile
  bar(ret);    // OK

  if (FOO(x))  // ERROR, may not compile
  if (ret)     // OK
.ve

  Example usage:
.vb
  #define MY_SIMPLE_RETURNING_MACRO(arg1) PetscMacroReturns(0,arg1+=10)

  int x = 10;
  int err = MY_SIMPLE_RETURNING_MACRO(x); // err = 0, x = 20

  // multiline macros allowed, but must declare with line continuation as usual
  #define MY_COMPLEX_RETURNING_MACRO(arg1) PetscMacroReturns(0, \
    if (arg1 > 10) {                                            \
      puts("big int!");                                         \
    } else {                                                    \
      return 7355608;                                           \
    }                                                           \
  )

  // if retexpr contains commas, must enclose it with braces
  #define MY_COMPLEX_RETEXPR_MACRO_1() PetscMacroReturns(x+=10,0,body...)
  #define MY_COMPLEX_RETEXPR_MACRO_2() PetscMacroReturns((x+=10,0),body...)

  int x = 10;
  int y = MY_COMPLEX_RETEXPR_MACRO_1(); // ERROR, y = x = 20 not 0
  int z = MY_COMPLEX_RETEXPR_MACRO_2(); // OK, y = 0, x = 20
.ve

.seealso: `PetscExpand()`, `PetscConcat()`, `PetscStringize()`
M*/
#define PetscMacroReturns(retexpr, ...) PetscMacroReturns_(retexpr, __VA_ARGS__)

#define PetscMacroReturnStandard(...) PetscMacroReturns(PETSC_SUCCESS, __VA_ARGS__)

/*MC
  PETSC_STATIC_ARRAY_LENGTH - Return the length of a static array

  Synopsis:
  #include <petscmacros.h>
  size_t PETSC_STATIC_ARRAY_LENGTH(a)

  Input Parameter:
. a - a static array of any type

  Output Parameter:
. <return-value> -  the length of the array

  Example:
.vb
  PetscInt a[22];
  size_t sa = PETSC_STATIC_ARRAY_LENGTH(a)
.ve
  `sa` will have a value of 22

  Level: intermediate
M*/
#define PETSC_STATIC_ARRAY_LENGTH(a) (sizeof(a) / sizeof((a)[0]))

/*
  These macros allow extracting out the first argument or all but the first argument from a macro __VAR_ARGS__ INSIDE another macro.

  Example usage:

  #define mymacro(obj,...) {
    PETSC_FIRST_ARG((__VA_ARGS__,unused));
    f(22 PETSC_REST_ARG(__VA_ARGS__));
  }

  Note you add a dummy extra argument to __VA_ARGS__ and enclose them in an extra set of () for PETSC_FIRST_ARG() and PETSC_REST_ARG(__VA_ARGS__) automatically adds a leading comma only if there are additional arguments

  Reference:
  https://stackoverflow.com/questions/5588855/standard-alternative-to-gccs-va-args-trick
*/
#define PETSC_FIRST_ARG_(N, ...)                                                                      N
#define PETSC_FIRST_ARG(args)                                                                         PETSC_FIRST_ARG_ args
#define PETSC_SELECT_16TH(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, ...) a16
#define PETSC_NUM(...)                                                                                PETSC_SELECT_16TH(__VA_ARGS__, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, ONE, throwaway)
#define PETSC_REST_HELPER_TWOORMORE(first, ...)                                                       , __VA_ARGS__
#define PETSC_REST_HELPER_ONE(first)
#define PETSC_REST_HELPER2(qty, ...) PETSC_REST_HELPER_##qty(__VA_ARGS__)
#define PETSC_REST_HELPER(qty, ...)  PETSC_REST_HELPER2(qty, __VA_ARGS__)
#define PETSC_REST_ARG(...)          PETSC_REST_HELPER(PETSC_NUM(__VA_ARGS__), __VA_ARGS__)

#define PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN_(name, ...) \
  _Pragma(PetscStringize(name diagnostic push)) \
  _Pragma(PetscStringize(name diagnostic ignored __VA_ARGS__))

#define PETSC_PRAGMA_DIAGNOSTIC_IGNORED_END_(name) _Pragma(PetscStringize(name diagnostic pop))

#if defined(__clang__)
  #define PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN(...) PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN_(clang, __VA_ARGS__)
  #define PETSC_PRAGMA_DIAGNOSTIC_IGNORED_END()      PETSC_PRAGMA_DIAGNOSTIC_IGNORED_END_(clang)
#elif defined(__GNUC__) || defined(__GNUG__)
  #define PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN(...) PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN_(GCC, __VA_ARGS__)
  #define PETSC_PRAGMA_DIAGNOSTIC_IGNORED_END()      PETSC_PRAGMA_DIAGNOSTIC_IGNORED_END_(GCC)
#endif

#ifndef PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN
  #define PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN(...)
  #define PETSC_PRAGMA_DIAGNOSTIC_IGNORED_END(...)
  // only undefine these if they are not used
  #undef PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN_
  #undef PETSC_PRAGMA_DIAGNOSTIC_IGNORED_END_
#endif

/* OpenMP support */
#if defined(_OPENMP)
  #if defined(_MSC_VER)
    #define PetscPragmaOMP(...) __pragma(__VA_ARGS__)
  #else
    #define PetscPragmaOMP(...) _Pragma(PetscStringize(omp __VA_ARGS__))
  #endif
#endif

#ifndef PetscPragmaOMP
  #define PetscPragmaOMP(...)
#endif

/* PetscPragmaSIMD - from CeedPragmaSIMD */
#if defined(__NEC__)
  #define PetscPragmaSIMD _Pragma("_NEC ivdep")
#elif defined(__INTEL_COMPILER) && !defined(_WIN32)
  #define PetscPragmaSIMD _Pragma("vector")
#elif defined(__GNUC__)
  #if __GNUC__ >= 5 && !defined(__PGI)
    #define PetscPragmaSIMD _Pragma("GCC ivdep")
  #endif
#elif defined(_OPENMP) && _OPENMP >= 201307
  #define PetscPragmaSIMD PetscPragmaOMP(simd)
#elif defined(PETSC_HAVE_CRAY_VECTOR)
  #define PetscPragmaSIMD _Pragma("_CRI ivdep")
#endif

#ifndef PetscPragmaSIMD
  #define PetscPragmaSIMD
#endif

#endif /* PETSC_PREPROCESSOR_MACROS_H */
