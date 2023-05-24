#ifndef PETSC_CPP_MACROS_HPP
#define PETSC_CPP_MACROS_HPP

#include <petscmacros.h>

#if defined(__cplusplus)

  // basic building blocks
  #define PETSC_DECLTYPE_AUTO(...) ->decltype(__VA_ARGS__)
  #define PETSC_NOEXCEPT_AUTO(...) noexcept(noexcept(__VA_ARGS__))
  #define PETSC_RETURNS(...) \
    { \
      return __VA_ARGS__; \
    }

  // one without the other
  #define PETSC_DECLTYPE_AUTO_RETURNS(...) PETSC_DECLTYPE_AUTO(__VA_ARGS__) PETSC_RETURNS(__VA_ARGS__)
  #define PETSC_NOEXCEPT_AUTO_RETURNS(...) PETSC_NOEXCEPT_AUTO(__VA_ARGS__) PETSC_RETURNS(__VA_ARGS__)

  // both
  #define PETSC_DECLTYPE_NOEXCEPT_AUTO(...) PETSC_NOEXCEPT_AUTO(__VA_ARGS__) PETSC_DECLTYPE_AUTO(__VA_ARGS__)
  // all
  #define PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(...) PETSC_DECLTYPE_NOEXCEPT_AUTO(__VA_ARGS__) PETSC_RETURNS(__VA_ARGS__)

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
  #define PETSC_CXX_COMPAT_DECL(...) PETSC_NODISCARD static inline __VA_ARGS__ noexcept

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

#endif // __cplusplus

#endif // PETSC_CPP_MACROS_HPP
