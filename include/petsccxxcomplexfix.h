#pragma once

/* MANSEC = Sys */

/*
    The pragma below silence all compiler warnings coming from code in this header file.
    In particular, it silences `-Wfloat-equal` warnings in `operator==()` and `operator!=` below.
    Other compilers beyond GCC support this pragma.
*/
#if defined(__GNUC__) && (__GNUC__ >= 4) && !defined(__NEC__)
  #pragma GCC system_header
#endif

/*
     Defines additional operator overloading for the C++ complex class that are "missing" in the standard
     include files. For example, the code fragment

     std::complex<double> c = 22.0;
     c = 11 + c;

     will produce a compile time error such as

     error: no match for 'operator+' (operand types are 'int' and 'std::complex<double>')

     The code fragment

     std::complex<float> c = 22.0;
     c = 11.0 + c;

     will produce a compile time error such as

     error: no match for 'operator+' (operand types are 'double' and 'std::complex<float>')

     This deficiency means one may need to write cumbersome code while working with the C++ complex classes.

     This include file defines a few additional operator overload methods for the C++ complex classes to handle
     these cases naturally within PETSc code.

     This file is included in petscsystypes.h when feasible. In the small number of cases where these additional methods
     may conflict with other code one may add '#define PETSC_SKIP_CXX_COMPLEX_FIX 1' before including any PETSc include
     files to prevent these methods from being provided.
*/

// In PETSc, a quad precision PetscComplex is a C type even with clanguage=cxx, therefore no C++ operator overloading needed for it.
#if !defined(PETSC_USE_REAL___FLOAT128)
  #include <type_traits>
// For operations "Atype op Cmplex" or "Cmplex op Atype" with Cmplex being PetscComplex, the built-in support allows Atype to be PetscComplex or PetscReal.
// We extend Atype to other C++ arithmetic types, and __fp16, __float128 if available.
// We put Cmplex as a template parameter so that we can enforce Cmplex to be PetscComplex and forbid compilers to convert other types to PetscComplex.
// This requires C++11 or later.
template <typename Cmplex, typename Atype> // operation on a complex and an arithmetic type
struct petsccomplex_extended_type :
  std::integral_constant<bool, (std::is_same<Cmplex, PetscComplex>::value && std::is_arithmetic<Atype>::value && !std::is_same<Atype, PetscReal>::value)
  #if defined(PETSC_HAVE_REAL___FP16) && !defined(PETSC_USE_REAL___FP16)
                                 || std::is_same<Atype, __fp16>::value
  #endif
  #if defined(PETSC_HAVE_REAL___FLOAT128) && !defined(PETSC_USE_REAL___FLOAT128)
                                 || std::is_same<Atype, __float128>::value
  #endif
                         > {
};

template <typename Cmplex, typename Atype>
inline typename std::enable_if<petsccomplex_extended_type<Cmplex, Atype>::value, Cmplex>::type operator+(const Atype &lhs, const Cmplex &rhs)
{
  return PetscReal(lhs) + rhs;
}

template <typename Cmplex, typename Atype>
inline typename std::enable_if<petsccomplex_extended_type<Cmplex, Atype>::value, Cmplex>::type operator+(const Cmplex &lhs, const Atype &rhs)
{
  return lhs + PetscReal(rhs);
}

template <typename Cmplex, typename Atype>
inline typename std::enable_if<petsccomplex_extended_type<Cmplex, Atype>::value, Cmplex>::type operator-(const Atype &lhs, const Cmplex &rhs)
{
  return PetscReal(lhs) - rhs;
}

template <typename Cmplex, typename Atype>
inline typename std::enable_if<petsccomplex_extended_type<Cmplex, Atype>::value, Cmplex>::type operator-(const Cmplex &lhs, const Atype &rhs)
{
  return lhs - PetscReal(rhs);
}

template <typename Cmplex, typename Atype>
inline typename std::enable_if<petsccomplex_extended_type<Cmplex, Atype>::value, Cmplex>::type operator*(const Atype &lhs, const Cmplex &rhs)
{
  return PetscReal(lhs) * rhs;
}

template <typename Cmplex, typename Atype>
inline typename std::enable_if<petsccomplex_extended_type<Cmplex, Atype>::value, Cmplex>::type operator*(const Cmplex &lhs, const Atype &rhs)
{
  return lhs * PetscReal(rhs);
}

template <typename Cmplex, typename Atype>
inline typename std::enable_if<petsccomplex_extended_type<Cmplex, Atype>::value, Cmplex>::type operator/(const Atype &lhs, const Cmplex &rhs)
{
  return PetscReal(lhs) / rhs;
}

template <typename Cmplex, typename Atype>
inline typename std::enable_if<petsccomplex_extended_type<Cmplex, Atype>::value, Cmplex>::type operator/(const Cmplex &lhs, const Atype &rhs)
{
  return lhs / PetscReal(rhs);
}

template <typename Cmplex, typename Atype>
inline typename std::enable_if<petsccomplex_extended_type<Cmplex, Atype>::value, bool>::type operator==(const Atype &lhs, const Cmplex &rhs)
{
  return rhs.imag() == PetscReal(0) && rhs.real() == PetscReal(lhs);
}

template <typename Cmplex, typename Atype>
inline typename std::enable_if<petsccomplex_extended_type<Cmplex, Atype>::value, bool>::type operator==(const Cmplex &lhs, const Atype &rhs)
{
  return lhs.imag() == PetscReal(0) && lhs.real() == PetscReal(rhs);
}

template <typename Cmplex, typename Atype>
inline typename std::enable_if<petsccomplex_extended_type<Cmplex, Atype>::value, bool>::type operator!=(const Atype &lhs, const Cmplex &rhs)
{
  return rhs.imag() != PetscReal(0) || rhs.real() != PetscReal(lhs);
}

template <typename Cmplex, typename Atype>
inline typename std::enable_if<petsccomplex_extended_type<Cmplex, Atype>::value, bool>::type operator!=(const Cmplex &lhs, const Atype &rhs)
{
  return lhs.imag() != PetscReal(0) || lhs.real() != PetscReal(rhs);
}

#endif
